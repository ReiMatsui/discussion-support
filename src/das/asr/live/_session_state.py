"""main()内の共有状態を集約するコンテナ."""
from __future__ import annotations

import collections
import contextlib
import datetime
import json
import os
import queue
import re
import struct
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from ._cluster_naming import ClusterVoiceNamer
from ._constants import (
    _MANUAL_CALL_MAX_CHARS,
    _PROACTIVITY_DEFAULT,
    _PROACTIVITY_PROFILES,
    AGENT_VOICES,
    CLEAR_LINE,
    DIM,
    HTML_PALETTE,
    HTML_TMPL,
    PALETTE,
    PYANNOTE_PARTICIPANT_HYSTERESIS_S,
    RESET,
    SR,
    UNSURE_SPEAKER,
    fmt_ts,
)
from ._diarization import DiarizationEvent, DiarizationProvider, SpeakerResolver
from ._participation import participation_stats
from ._seat_audio import RetroAttributor, SeatAudio
from ._speaker_keys import (
    NON_PARTICIPANT_KEYS,
    is_provisional_key,
    looks_like_system_name,
)
from ._voice_profiles import VoiceProfiles
from .agents._partner import ConversationPartner
from .agents._realtime import RealtimeAgent
from .agents._simulator import DiscussionSimulator


class SessionState:
    """main()内の共有状態を集約するコンテナ.

    巨大だった main() のクロージャ変数をインスタンス属性に集約し、
    ヘルパーメソッドとして外部からアクセス可能にする。
    """

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def __init__(self, *, args, started, out_path, html_path, diag_path,
                 turns_path, wav_path, tracker=None, serve=True,
                 diarization_provider: DiarizationProvider | None = None,
                 speaker_resolver: SpeakerResolver | None = None,
                 cluster_namer: ClusterVoiceNamer | None = None,
                 seat_audio: SeatAudio | None = None):
        self.args = args
        self.stt_backend = None
        self.started = started
        self.out_path = out_path
        self.html_path = html_path
        self.diag_path = diag_path
        self.turns_path = turns_path
        self.interventions_path = self._interventions_path_for(turns_path)
        # 採否レビュー（Controller が実際に採択した判断のログ）。
        self.intervention_review_path = self._intervention_review_path_for(turns_path)
        self.wav_path = wav_path
        self._serve = serve

        # 発話記録
        self.names: dict[str, str] = {}
        self.colors: dict[str, str] = {}
        # HTML色の安定インデックス（review C11/P2）。従来は list(colors).index(key)
        # で色を決めていたため、rekey の pop で後続キー全員の色がずれた。
        # 単調増加の採番を一度割り当てたら変えない（統合時は旧キーの番号を引き継ぐ）。
        self.html_color_idx: dict[str, int] = {}
        self.html_color_seq = 0
        self.records: list[dict] = []
        self.state_lock = threading.Lock()
        # 会議世代カウンタ（H2）。reset_for_new_meeting が state_lock 内で +1 する。
        # 各 worker はスナップショット取得時に epoch を読み、副作用（feed/キュー投入/
        # カーソル書き戻し）の直前に一致を再確認する。リセットを跨いだ古い計算結果を
        # 新会議に書き込んで「カーソルが発話数を超える永久待機」に陥るのを防ぐ。
        self.meeting_epoch = 0

        # 声紋
        self.tracker: VoiceProfiles | None = tracker
        # 外部話者分離
        self.diarization_provider = diarization_provider
        self.speaker_resolver = speaker_resolver or SpeakerResolver()
        self.diarization_events: list[DiarizationEvent] = []
        self.diarization_lock = threading.Lock()
        self.diarization_speaker_keys: dict[str, str] = {}
        # 参加者化ヒステリシス用: まだ @diar:N を発行していないラベルの累積発話ms。
        # pyannote provider 使用時のみ参照される（他providerは従来どおり即時登録）。
        self.diarization_pending_ms: dict[str, int] = {}
        # @diar:N の採番用の単調増加カウンタ。クラスタ間名寄せは
        # diarization_speaker_keys の吸収側エントリを pop して辞書を縮めるため、
        # len ベースの採番だと使用中の @diar:N を別人へ再発行してしまう
        # （docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照）。
        self.diarization_key_seq = 0
        # ハイブリッド構成（docs/design/pyannote_live1_trial_2026-07-09.md §9）:
        # pyannoteクラスタ単位の声紋照合による名前付け。--vp-cluster-naming
        # 指定時のみ _bootstrap.py が生成して渡す。None なら従来どおり
        # key_for_diarization_speaker の匿名キー付与のみで完結する。
        self.cluster_namer: ClusterVoiceNamer | None = cluster_namer
        # 席落ち発話の割当て（handoff §27。ハイブリッド構成でのみ生成される）。
        # None なら従来どおり席上限で落ちた発話は未確定のまま。
        self.seat_audio: SeatAudio | None = seat_audio
        # 序盤に決めた帰属の貼り直し（handoff §28）。seat_audio が無い構成では
        # 生成されないので、pyannote単独・Soniox単独は従来どおり見直さない。
        self.retro: RetroAttributor | None = (
            RetroAttributor(seat_audio) if seat_audio is not None else None)
        self.anonymous_labels: dict[str, str] = {}
        self._DIARIZATION_KEEP_MS = 10 * 60 * 1000
        # 上限(constrain)で未確定化した回数（可視化用, 2026-07-25 実セッションで
        # 「上限1のまま2人会話→2人目が全滅」が無警告で起きた対策）。帰属は変えない。
        self.constrain_drop_counts: dict[str, int] = {}
        self.constrain_warned = False

        # AI
        self.agent: RealtimeAgent | None = None
        self.partner: ConversationPartner | None = None
        self.simulator: DiscussionSimulator | None = None
        # 会話モード(converse)でパートナーを動的生成するための設定（F3）。
        # bootstrapで {api_key, voice, topic} をセットする。
        self._partner_cfg: dict | None = None

        # 論点
        self.topics: list[dict] = []
        self.topics_lock = threading.Lock()
        self.topic_cursor = 0
        self._TOPIC_WINDOW = 10
        self._TOPIC_TRIGGER = 5
        self.drift_cursor = 0
        self.fact_cursor = 0
        self.triage_cursor = 0
        # 脱線検出→介入トリガーの受け渡しキュー（R2: トリガー経路の単一化）。
        # _run_drift_checker が積み、_run_agent_worker が裁定して trigger する。
        self.drift_requests: queue.Queue[str] = queue.Queue()
        # 参加度の声かけ要求キュー（S4）。対象話者の表示名を積む。
        self.invite_requests: queue.Queue[str] = queue.Queue()
        # 事実誤りの短い補正要求キュー。{"correction": str, ...} を積む。
        self.factcheck_requests: queue.Queue[dict] = queue.Queue()
        # 手動呼び出しキュー（Phase1）。UI/音声から明示的にファシリテーターを呼ぶ。
        # {"request": str, "source": "ui"|"voice", "created_at": monotonic} を積む。
        self.manual_call_requests: queue.Queue[dict] = queue.Queue()
        # 整理介入（summarize）の要求キュー（C3）。_run_structuring_checker が
        # 「今、整理が価値を足す」とLLM判定したときだけ {"focus": str} を積む。
        # _run_agent_worker が裁定して trigger する（count の無条件介入を置換）。
        self.summarize_requests: queue.Queue[dict[str, Any]] = queue.Queue()
        # AF ベース介入の要求キュー（H1 フェーズ4）。_run_af_checker が
        # decide_intervention + 価値ゲートの結果を積む。AF ランタイム有効時のみ動く。
        # {"kind": "af_l1"|"af_l2", "brief", "af_text", "target_speaker"} を積む。
        self.af_requests: queue.Queue[dict[str, Any]] = queue.Queue()
        # AF ランタイム (run_af_runtime がセット)。None ならルールベースのみ。
        self.af_runtime: Any = None
        # 直近の手動呼び出しの進行状況（UX観測用, UI表示）。
        # {"status": queued|waiting|dispatched|delivered|expired|cancelled,
        #  "detail", "source", "request", "at", "wait_sec"} または None。
        self.manual_call_status: dict | None = None
        # 認識途中経過（partial）。UIに「認識中」を見せるため（課題①）。
        self.partial_text = ""
        self.partial_speaker = ""
        # partial が最後に変化した時刻（F3: フロア占有判定の stale ガード用）。
        self._last_partial_change = 0.0
        # ファシリテーター発言の副作用イベント（議事録追加・パートナー反応）。
        # agentの受信スレッドはここに積むだけにして、専用ワーカーが処理する。
        # 受信スレッドが partner の WebSocket 送信やファイルI/Oでブロックするのを防ぐ。
        self.fac_events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        # 積極性プロファイル（S5）。bootstrapで --proactivity から上書きされる。
        self.proactivity_name = _PROACTIVITY_DEFAULT
        self.proactivity: dict = dict(_PROACTIVITY_PROFILES[_PROACTIVITY_DEFAULT])
        self.intervention_enabled = True
        self.intervention_events: list[dict] = []
        self._intervention_event_seq = 0
        self._last_intervention_event_id: str | None = None
        self._last_intervention_event_reason: str | None = None
        # UIからの停止フック（F1）。run_sessionが「stopを立ててwsを閉じる」関数を設定する。
        self.request_stop: Callable[[], None] | None = None
        # 変更リビジョン（F2）。save()ごとに+1。SSEはこの変化を見て差分配信する。
        self.rev = 0
        # フルリセット（STT接続の作り直し）用。UIは request_reset を呼ぶだけで、
        # 実際の作り直しは ws を所有するメインスレッドが行う。
        self.stt_ws = None                       # 現在のSTT WebSocket（動的参照）
        self.reset_requested = threading.Event()  # メインスレッドへの作り直し要求
        self.resetting = False                   # UI表示用（リセット処理中）
        self.request_reset: Callable[[], None] | None = None
        self.waiting_to_start = False             # 開始前設定画面で待機中
        self.start_requested = threading.Event()

        # PCMバッファ
        self.pcm_buf = bytearray()
        self.pcm_buf_offset = 0
        self.pcm_total_bytes = 0
        # 声紋判定用: STT WebSocket に実際に送信できたPCMだけを保持する。
        # 録音用バッファとは分けることで、接続リセット中に捨てた音声や送信失敗音声で
        # STTタイムスタンプと声紋切り出し位置がずれるのを防ぐ。
        self.asr_pcm_buf = bytearray()
        self.asr_pcm_buf_offset = 0
        self.asr_pcm_total_bytes = 0
        self.stt_time_offset_ms = 0
        self._stt_connection_audio_base_bytes = 0
        self._PCM_KEEP_BYTES = SR * 2 * 120
        self.buf_lock = threading.Lock()
        self.pcm_file = None  # IO[bytes] | None
        # AI再生区間の記録（P2-1）。マイク音声のmsタイムラインで「AIが鳴っていた
        # 区間」を残し、エコー判定を壁時計ではなく発話区間の重なりで行う。STT確定が
        # 遅れてエコー窓の外に出た回り込みも取りこぼさない。source ごとに開いた区間を
        # 保持し、終了時に閉じて deque に積む。
        self._ai_speech_lock = threading.Lock()
        self._ai_speech_intervals: collections.deque[tuple[int, int, str]] = (
            collections.deque(maxlen=64))
        self._ai_speech_open: dict[str, int] = {}
        # パートナー切断後もエコー参照を短時間保持する（P2-4）。detach で partner が
        # None になるとテキストエコー防御が即消えるため、退役直前の応答テキストを
        # TTL 内だけ照合対象に残す。
        self._RETIRED_ECHO_TTL = 10.0
        self.retired_echo_texts: collections.deque[tuple[float, str]] = (
            collections.deque(maxlen=32))

        # 制御
        self.stop = threading.Event()
        self.audio_q: queue.Queue[bytes | None] = queue.Queue()

        # エージェントワーカー状態
        self._last_utt_time = [time.monotonic()]
        self._was_in_echo = [False]
        self.agent_cursor = 0

    # ------------------------------------------------------------------
    # 表示ヘルパー
    # ------------------------------------------------------------------
    @staticmethod
    def _interventions_path_for(turns_path: str) -> str:
        if turns_path.endswith(".turns.jsonl"):
            return turns_path[:-len(".turns.jsonl")] + ".interventions.jsonl"
        if turns_path.endswith(".jsonl"):
            return turns_path[:-len(".jsonl")] + ".interventions.jsonl"
        return turns_path + ".interventions.jsonl"

    @staticmethod
    def _intervention_review_path_for(turns_path: str) -> str:
        if turns_path.endswith(".turns.jsonl"):
            return turns_path[:-len(".turns.jsonl")] + ".intervention_review.jsonl"
        if turns_path.endswith(".jsonl"):
            return turns_path[:-len(".jsonl")] + ".intervention_review.jsonl"
        return turns_path + ".intervention_review.jsonl"

    def disp_name(self, key) -> str:
        key = str(key)
        if key == UNSURE_SPEAKER:
            return "未確定"
        name = self.names.get(key)
        if name and not looks_like_system_name(name):
            return name
        if is_provisional_key(key) or looks_like_system_name(key) or name:
            return self._anonymous_label_for(key)
        return key

    def peek_disp_name(self, key) -> str:
        """割当てを行わない読み取り専用の表示名取得.

        disp_name は未割当ての匿名キーに新しいラベル文字を割り当てる副作用を持つ。
        partial（認識途中）表示のためだけに呼ぶと、flush まで到達しない一時的な
        STTラベル（幻キー）がラベル文字とスロットを恒久的に消費してしまう
        （2026-07-14 実セッション: 議事録に一度も現れない幻キーが「参加者B」を
        占有、docs/design/handoff_2026-07-14_unregistered_speakers.md 参照）。
        表示のみの経路（show_partial / vp_debug の途中経過出力）はこちらを使う。
        未割当てなら中立表示「?」を返し、辞書は変更しない。
        """
        key = str(key)
        if key == UNSURE_SPEAKER:
            return "未確定"
        name = self.names.get(key)
        if name and not looks_like_system_name(name):
            return name
        if is_provisional_key(key) or looks_like_system_name(key) or name:
            return self.anonymous_labels.get(key, "?")
        return key

    @staticmethod
    def _anonymous_suffix(index: int) -> str:
        letters = ""
        n = index
        while True:
            n, rem = divmod(n, 26)
            letters = chr(ord("A") + rem) + letters
            if n == 0:
                return letters
            n -= 1

    def _anonymous_label_for(self, key: str) -> str:
        if key not in self.anonymous_labels:
            # 「累積数」ではなく「未使用の最小文字」を割り振る。幻の話者キー
            # （AI回り込み・重なり由来の一時キー）が統合で消えれば、その文字は次の
            # 新規話者が再利用でき、連番の飛び・len ベースの重複が構造的に消える。
            used = set(self.anonymous_labels.values())
            i = 0
            while True:
                label = f"参加者{self._anonymous_suffix(i)}"
                if label not in used:
                    self.anonymous_labels[key] = label
                    break
                i += 1
        return self.anonymous_labels[key]

    def _displays_real_name(self, key: str) -> bool:
        """key が実名（話者N/人物N でも #/@diar 匿名キーでもない名前）で表示されるか."""
        name = self.names.get(key)
        if name and not looks_like_system_name(name):
            return True
        return not (is_provisional_key(key)
                    or looks_like_system_name(key))

    def set_display_name(self, key: str, name: str) -> None:
        """表示名を設定する。実名なら匿名ラベルの文字を解放する（リネームの共通経路）.

        実名が付いたキーは以後その名前で表示されるため文字は不要。解放すれば後続の
        新規参加者が若い文字を再利用できる（連番の飛びの解消）。
        """
        with self.state_lock:
            self.names[key] = name
            if not looks_like_system_name(name):
                self.anonymous_labels.pop(key, None)

    def _max_human_speakers(self) -> int | None:
        value = getattr(self.args, "diarization_max_speakers", None)
        return value if isinstance(value, int) and value > 0 else None

    def _human_slot_key(self, key: str) -> str | None:
        if key in NON_PARTICIPANT_KEYS:
            return None
        if is_provisional_key(key) or looks_like_system_name(key):
            return self.anonymous_labels.get(key, key)
        return key

    def _known_human_slots(self) -> set[str]:
        """既に席を持っている人の集合を返す（表示ラベル、または実名キー）.

        records には話者を持たないレコードも混ざる（add_sys のシステム
        メッセージ: 「この声を『参加者A』として追跡開始」「注意: 想定話者数の
        上限…」など）。これらを話者キー "" として数えると、**空文字が1つの席
        として居座り、想定話者数が実質1人ぶん目減りする**（2026-07-25 に統一席
        ルールで records 走査を導入した際の取りこぼし。sys は鋳造のたびに必ず
        出るので、上限2の2人会話では実際に座れるのが1人だけになっていた）。
        話者を持つレコードだけを数える。
        """
        slots = set(self.anonymous_labels.values())
        with self.state_lock:
            for r in self.records:
                if "speaker" not in r:
                    continue
                slot = self._human_slot_key(str(r["speaker"]))
                if slot:
                    slots.add(slot)
        return slots

    def constrain_human_speaker_key(self, key) -> str:
        """参加人数上限を超える新規話者を「未確定」に落とす（統一席ルール）.

        ルールは一文: **既に席（表示ラベル）を持つ人は常に通す。新しい人は
        席が空いていれば入れる。満席なら未確定＋警告**。キーの種別
        （@diar:N / #N / 人物N）による例外は持たない（2026-07-25 一掃。
        経緯は関数内コメント参照）。

        名簿を確定（closed roster, tracker.auto == False）している場合はこれより優先し、
        「登録済みのアクティブな人 or 未確定」だけを許す。声紋以外の経路（外部
        diarization の @diar:N や STT フォールバック）で作られた匿名キーも含めて
        未確定に落とし、参加人数(diarization_max_speakers)には依存しない。
        """
        key = str(key)
        if key in NON_PARTICIPANT_KEYS:
            return key
        tracker = self.tracker
        if tracker is not None and not getattr(tracker, "auto", True):
            roster = set(tracker.active_profile_names())
            return key if key in roster else UNSURE_SPEAKER
        # 統一ルール（2026-07-25 一掃）: 「既に席（表示ラベル）を持つ人は常に
        # 通す。新しい人は席が空いていれば入れる。満席なら未確定＋警告」。
        #
        # 旧実装は2つの帳簿を併用していた:
        #   (a) 声紋アクティブ人物（人物N/実名）の無条件素通し
        #   (b) 匿名キーのラベル文字辞書順スロット選別（sorted(labels)[:max]）
        # (a) は「上限2なのに参加者Cが現れる」（声紋の二重登録が3人目として表示、
        # 2026-07-25 実セッション）を生み、(b) は「文字の巡り合わせだけで既存
        # 人物が全滅」（2026-07-14 実セッション、人物2=参加者D事故）を生んだ。
        # 統一ルールは両事故を同時に防ぐ: 席を持つ人物は文字順に関係なく通り
        # （→2026-07-14 の保護は素通し無しで成立）、新規は種別（@diar/#/人物N）
        # を問わず席が無ければ入れない（→参加者Cは生まれず未確定+警告になる。
        # 声紋の二重登録された人物の発話は、合流(dedupe)か上限の引き上げで回収）。
        max_speakers = self._max_human_speakers()
        if max_speakers is None:
            return key
        slots = self._known_human_slots()
        slot = self._human_slot_key(key)
        if slot is not None and slot in slots:
            return key   # 既に席を持つ（表示ラベル持ち、または今回喋った実名）
        # 実名キーも席を要求する（2026-07-26 修正）。以前は「実名リネーム済みの
        # 人物は既存の席保持者」として無条件に通していたが、voices.json は
        # **セッションをまたいで**プロファイルを保持するため、過去の会議で
        # 登録された人物（聞き間違いの「壁」「朱色」等を含む）が今回まだ一度も
        # 喋っていないのに素通りし、上限3の会話で8人が表示された（実会話で確認）。
        # 「今回のセッションで既に席に着いているか」で判定するのが正しい。
        if len(slots) >= max_speakers:
            self._note_constrain_drop(key, max_speakers)
            return UNSURE_SPEAKER
        self._note_seat_admitted(key, slots, max_speakers)
        return key

    def _note_seat_admitted(self, key: str, slots: set, max_speakers: int) -> None:
        """新しく席に着いた瞬間を diag に残す（診断のみ・挙動不変）.

        2026-07-27 の実会話で「上限3なのに4人目が席に着いた」が起きたが、
        ロジック単体では正しく落ちる（`constrain_human_speaker_key` に同じ
        並びを直接与えると4人目は未確定になる）。ライブ特有の経路——
        rekey で新しいキーへ移った直後、遡及訂正のラベル詰め直しの直後など
        ——で席の集合がどう見えていたかを記録しないと原因が絞れない。
        記録するのは「そのとき席をどう数えたか」そのもの。
        """
        with contextlib.suppress(OSError), \
                open(self.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "seat_admitted", "key": key,
                "slots": sorted(str(x) for x in slots),
                "max_speakers": max_speakers,
                "labels": dict(self.anonymous_labels),
            }, ensure_ascii=False, default=str) + "\n")

    def _note_constrain_drop(self, key: str, max_speakers: int) -> None:
        """上限で未確定化した事実を可視化する（帰属の挙動は一切変えない）.

        2026-07-25 の実セッションで「想定話者数1のまま2人会話→2人目の
        クラスタ(31発話)が無警告で全滅」が起きた対策。diag に構造化イベントを
        残し、同一キーが繰り返し落ちる場合は一度だけ議事録/UIに警告を出す。
        """
        n = self.constrain_drop_counts.get(key, 0) + 1
        self.constrain_drop_counts[key] = n
        with contextlib.suppress(OSError), \
                open(self.diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "constrain_drop", "key": key, "count": n,
                "max_speakers": max_speakers,
                "slots": sorted(set(self.anonymous_labels.values())),
            }, ensure_ascii=False) + "\n")
        if n == 3 and not self.constrain_warned:
            self.constrain_warned = True
            self.add_sys(self.elapsed_ms(),
                         f"注意: 想定話者数の上限（{max_speakers}人）により、"
                         "新しい話者の発話を未確定にしています。実際の参加人数が"
                         "多い場合は「想定話者数」を増やしてください")
            self.rev += 1

    def elapsed_ms(self) -> int:
        """セッション開始からの経過ミリ秒（sysメッセージのタイムスタンプ用）."""
        return max(0, int((datetime.datetime.now() - self.started)
                          .total_seconds() * 1000))

    def _uses_pyannote_hysteresis(self) -> bool:
        """新規ラベルの参加者化にヒステリシスを適用するか.

        pyannote Live-1 はセッション序盤にラベルが揺れ、一人の発話が
        SPEAKER_00/01/02... に分裂することが実地の検証（Live-1試行,
        docs/design/pyannote_live1_trial_2026-07-09.md）で確認された。
        即座に @diar:N を発行すると偽参加者が量産されるため、pyannote
        provider使用時のみ猶予を設ける。Soniox/AssemblyAI経路の挙動は
        変えない（従来どおり即時登録）。

        sortformer（ローカルSortformer, 2026-07-22 追加）もクラスタ型の
        匿名ラベル（SPEAKER_00..03）を出す同族であり、序盤の揺れ・短い
        誤活性に同じリスクがあるため、同じ猶予を適用する（誤帰属より
        未確定を選ぶ設計原則にも整合）。
        """
        provider = self.diarization_provider
        return provider is not None and getattr(provider, "name", None) in (
            "pyannote", "sortformer")

    def key_for_diarization_speaker(
        self, source: str, speaker: str, duration_ms: int = 0
    ) -> str:
        """外部diarizationの生ラベルを、表示用の安定した内部キーに変換する.

        pyannote の ``SPEAKER_00`` は実名でもUI向けラベルでもなく、provider内部の
        未登録クラスタIDにすぎない。recordsには内部キーを入れ、表示は参加者A/Bに統一する。

        ``duration_ms`` はこの発話の長さ（ms）。pyannote provider使用時のみ、
        同一の生ラベル（source:speaker）の累積発話が
        ``PYANNOTE_PARTICIPANT_HYSTERESIS_S`` 秒に達するまでは @diar:N を
        発行せず UNSURE_SPEAKER を返す（未確定のまま扱う）。達した時点で
        @diar:N を新規発行するが、それまでに UNSURE_SPEAKER として記録済みの
        過去レコードの遡及的な付け替えは行わない（records は speaker_source /
        diarization_raw_speaker で追跡可能なため事後分析は可能だが、
        RecvLoop.flush() の二段階構成 [resolve→append] をまたぐ安全な
        rename配線が低コストにできなかったため今回は見送り。未確定のまま
        UIに残る）。
        """
        raw = f"{source}:{speaker}"
        # 台帳 diarization_speaker_keys は recvスレッドが書き、rekey（UIの
        # /rename・/activate 経由＝別スレッド）が state_lock 下で走査・書き換えする。
        # 発行側がロックを取らないと、rekey の .items() 走査中に新キーが挿入されて
        # 「dictionary changed size during iteration」で rekey が落ちる
        # （2026-07-25 監査）。読み取り→採番→登録を1手にまとめて守る。
        with self.state_lock:
            if raw in self.diarization_speaker_keys:
                return self.diarization_speaker_keys[raw]
            if self._uses_pyannote_hysteresis():
                threshold_ms = int(PYANNOTE_PARTICIPANT_HYSTERESIS_S * 1000)
                pending = self.diarization_pending_ms.get(raw, 0) + max(0, duration_ms)
                if pending < threshold_ms:
                    self.diarization_pending_ms[raw] = pending
                    return UNSURE_SPEAKER
                self.diarization_pending_ms.pop(raw, None)
            # len ベースの採番は名寄せの pop で辞書が縮むと使用中キーを再発行するため、
            # 単調増加カウンタで採番する（名寄せ無しなら len+1 と同値＝従来挙動不変）。
            self.diarization_key_seq += 1
            key = f"@diar:{self.diarization_key_seq}"
            self.diarization_speaker_keys[raw] = key
            return key

    def key_for_stt_fallback_speaker(self, speaker: str, duration_ms: int = 0) -> str:
        """外部diarizationが薄い時のSTTラベルも表示用の内部キーへ正規化する.

        注意: ``_uses_pyannote_hysteresis`` は provider 名しか見ないため、
        pyannote 使用時は STT フォールバックラベル（"stt:N"）にも3秒ヒステリシスが
        掛かる。当初の設計意図（pyannote のラベル揺れ対策）とラベルの出所は
        一致していないが、「diarization が瞬断した数秒間の STT ラベルが即座に
        新規参加者化するのを防ぐ」効果があり安全側なので、現状は仕様として
        維持する（docs/design/attribution_logic_review_2026-07.md C7。
        P1 の状態機械統合で再設計する際に扱いを一本化する）。
        """
        return self.key_for_diarization_speaker("stt", speaker, duration_ms)

    def key_for_label(self, sp) -> str:
        if str(sp).strip().lower() in {"", "none", "null", "unknown", "uu", UNSURE_SPEAKER}:
            return UNSURE_SPEAKER
        sp = str(sp)
        if self.tracker is not None and sp in self.tracker.sp_map:
            return self.tracker.sp_map[sp]
        return "#" + sp

    def drain_diarization_provider(self) -> None:
        """外部diarization providerから届いた閉区間を取り込む."""
        provider = self.diarization_provider
        if provider is None:
            return
        events = provider.drain_events()
        if not events:
            return
        with self.diarization_lock:
            self.diarization_events.extend(events)
            newest = max((e.end_ms or e.start_ms for e in self.diarization_events),
                         default=0)
            cutoff = newest - self._DIARIZATION_KEEP_MS
            self.diarization_events = [
                e for e in self.diarization_events
                if (e.end_ms or e.start_ms) >= cutoff
            ]

    def diarization_window(self, start_ms: int | None, end_ms: int | None) -> list[DiarizationEvent]:
        """発話区間と重なる外部diarizationイベントを返す."""
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            return []
        self.drain_diarization_provider()
        active = getattr(self.diarization_provider, "active_events", None)
        active_events = (
            active()
            if self.diarization_provider is not None and active is not None else []
        )
        with self.diarization_lock:
            return [
                e for e in self.diarization_events
                if e.end_ms is not None
                and min(e.end_ms, end_ms) - max(e.start_ms, start_ms) > 0
            ] + [
                e for e in active_events
                if min(e.end_ms or end_ms, end_ms) - max(e.start_ms, start_ms) > 0
            ]

    def color_of(self, key) -> str:
        key = str(key)
        if key not in self.colors:
            self.colors[key] = PALETTE[len(self.colors) % len(PALETTE)]
        return self.colors[key]

    def html_color(self, key) -> str:
        """話者キーに対応する安定したHTML色（ブラウザ再読み込みでも一貫, 課題②）.

        色は初回登録時に採番する単調カウンタで決まる（サーバー側に保持）。
        再読み込みではサーバー状態が変わらないため色がぶれず、新しい会議で
        リセットされる。rekey（統合・リネーム）では旧キーの番号を引き継ぐため、
        他の話者の色もずれない（旧実装は list(colors).index() だったため、
        rekey の pop で後続キーの色が全部ずれた。review C11/P2）。
        """
        key = str(key)
        self.color_of(key)  # ANSI 色の登録順も従来どおり確定させておく
        if key not in self.html_color_idx:
            self.html_color_idx[key] = self.html_color_seq
            self.html_color_seq += 1
        return HTML_PALETTE[self.html_color_idx[key] % len(HTML_PALETTE)]

    def rekey(self, old: str, new: str):
        """表示キーの付け替え: records・色・台帳を一括移行する単一入口.

        話者IDの台帳は records/colors/names のほかに diarization_speaker_keys
        （外部クラスタ→表示キー）と ClusterVoiceNamer._confirmed（クラスタ→
        確定名）があり、従来はここが更新されず、UI /rename・stdin fix の後に
        旧キーがクラスタ経由で復活していた（docs/design/
        attribution_logic_review_2026-07.md C3）。リネーム・統合の全経路
        （UI /rename・/activate、stdin fix/enroll、クラスタ確定の遡及統合）は
        必ず本メソッドを通るため、伝搬はここに一元化する（P2）。
        """
        with self.state_lock:
            for r in self.records:
                if r.get("speaker") == old:
                    r["speaker"] = new
            # 外部diarizationクラスタの台帳。旧キーを指したままだと、以後の
            # そのクラスタの発話が旧キー（削除済みの人物N等）で帰属される。
            for cluster, k in list(self.diarization_speaker_keys.items()):
                if k == old:
                    self.diarization_speaker_keys[cluster] = new
            # クラスタ確定名。observe() が確定名を短絡で返すため、ここを
            # 更新しないと旧名が永久に返り続ける（C3 の人格復活の本体）。
            if self.cluster_namer is not None:
                self.cluster_namer.rename_confirmed(old, new)
            # 席の参照音声。追従しないとリネーム後も旧キーへ寄せてしまい、
            # 消えたはずの人格が復活する（cluster_namer と同じ事情）。
            if self.seat_audio is not None:
                self.seat_audio.rename(old, new)
            if old in self.html_color_idx:
                # 統合先が未採番なら旧キーの色番号を引き継ぐ（同一人物の色を維持
                # しつつ、他の話者の色をずらさない。review C11）。
                self.html_color_idx.setdefault(new, self.html_color_idx.pop(old))
            if old in self.colors:
                self.colors.setdefault(new, self.colors.pop(old))
            if old in self.names:
                # 表示名も colors と同じ流儀で移行する（new に既に名前があれば
                # old 側は捨てる）。残留させると古いキーの names が溜まり続ける。
                self.names.setdefault(new, self.names.pop(old))
            if old in self.anonymous_labels:
                if self._displays_real_name(new):
                    # 実名に統合された → old の文字を解放（引き継がない）。後続の
                    # 新規参加者がその文字を再利用でき、飛びを防ぐ。
                    self.anonymous_labels.pop(old, None)
                else:
                    # 匿名キー同士の統合（#label→人物N など）は従来どおり文字を引き継ぐ。
                    self.anonymous_labels.setdefault(
                        new, self.anonymous_labels.pop(old))

    def apply_retro_attribution(self, revised: dict) -> dict:
        """遡及訂正の結果を records に反映する（handoff §28）.

        対象は「席の音声で決めた発話」と「未確定のまま残った発話」だけで、
        声紋層が高信頼で判定した発話（``speaker_source`` がそれ以外）は
        触らない——そちらは席の平均より強いという §27.12 の判断を変えない。

        リネームとの関係: ユーザーが付けた名前は `rekey` が records の
        speaker キーごと移すので、ここで扱うのは常に「現在のキー」である。
        本メソッドは席から席へ移すだけで、名前の台帳（names/colors の対応）は
        変えない。

        戻り値: 実際に書き換えた ``{ms: 新しいキー}``（空なら呼び出し側は
        通知も保存もしない）。**diag に残すために内容ごと返す**——遡及訂正は
        過去のレコードを書き換えるので、発話ごとの diag 行（flush 時点の
        final_key）だけでは最終状態を復元できない（handoff §28.10）。
        """
        if not revised:
            return {}
        applied: dict = {}
        with self.state_lock:
            for r in self.records:
                if "speaker" not in r:
                    continue
                ms = r.get("ms")
                new_key = revised.get(ms)
                if new_key is None or str(r["speaker"]) == str(new_key):
                    continue
                src = r.get("speaker_source")
                revisable = (src == "seat_assign"
                             or str(r["speaker"]) == UNSURE_SPEAKER)
                if not revisable:
                    continue
                r["speaker"] = new_key
                r["speaker_source"] = "seat_assign_retro"
                r["unsure"] = False
                applied[ms] = str(new_key)
            if applied:
                for r in self.records:
                    if "speaker" in r:
                        self.color_of(r["speaker"])
        return applied

    def compact_anonymous_labels(self) -> int:
        """発言の無くなったキーが押さえている表示文字を解放し、詰め直す.

        **なぜ要るのか**: 表示ラベル（参加者A/B/…）は「未使用の最小文字」を
        キーに割り当てる。ところが席の割当てや遡及訂正で発話が別のキーへ
        移ると、**元のキーは1件も発話を持たないまま文字だけ押さえ続ける**。
        その結果、参加者が1人しか居ないのに「参加者B」から始まる、という
        気持ちの悪い表示になる（2026-07-26 に実会話で確認）。

        やること: records に1件も現れないキーを台帳から落とし、残ったキーへ
        **初出順**に文字を振り直す。初出順を保つので、既に見えている人同士の
        文字が入れ替わることはない（空いた文字へ順に詰まるだけ）。

        実名で表示されているキーは `rekey` の時点で台帳から抜けているため、
        ここでは扱わない（ユーザーが付けた名前には触れない）。

        戻り値: 変わった件数（0 なら呼び出し側は保存も通知もしない）。
        """
        with self.state_lock:
            first_seen: dict[str, int] = {}
            for i, r in enumerate(self.records):
                key = str(r.get("speaker", "")) if "speaker" in r else ""
                if key and key in self.anonymous_labels and key not in first_seen:
                    first_seen[key] = i
            ordered = sorted(first_seen, key=lambda k: first_seen[k])
            fresh = {k: f"参加者{self._anonymous_suffix(i)}"
                     for i, k in enumerate(ordered)}
            if fresh == self.anonymous_labels:
                return 0
            changed = sum(1 for k, v in fresh.items()
                          if self.anonymous_labels.get(k) != v)
            changed += sum(1 for k in self.anonymous_labels if k not in fresh)
            self.anonymous_labels = fresh
            return changed

    def add_sys(self, ms, text: str):
        """システムイベントを議事録のタイムラインに残す."""
        with self.state_lock:
            self.records.append({"ms": ms, "sys": text})

    # ------------------------------------------------------------------
    # UI(API)向けの状態スナップショット（F1）
    # ------------------------------------------------------------------
    def session_mode(self) -> str:
        """現在のセッションモード: transcribe / converse / facilitate.

        - transcribe : 議事録のみ（エージェントoff or 無し）
        - converse   : AIと会話（パートナー有り＋ファシリテーター）
        - facilitate : 人間同士に介入（ファシリテーター単体）
        """
        if self.agent is None or not self.agent.enabled:
            return "transcribe"
        if self.partner is not None:
            return "converse"
        return "facilitate"

    def _speaker_label(self, key: str) -> str | None:
        """話者リネーム(/rename)に渡すラベルを返す。リネーム不可なら None.

        登録できるのは「声紋で確定したが名前の付いていない参加者」だけ。
        - 内部キー "人物N" → そのキー（profilesへ直接命名）
        暫定の "#N"（声紋未確定・Sonioxラベル依存で別人に振り替わりうる）、命名済みの
        実名、AI、未確定(?) は登録対象外（None）。確定した人だけに名前を付けることで、
        まだ揺れている話者に誤って名前を固定してしまうのを防ぐ。
        """
        key = str(key)
        if key.startswith("人物"):
            return key
        return None

    def api_snapshot(self) -> dict:
        """UI(API)向けの現在状態（JSON化可能なdict）を返す."""
        with self.state_lock:
            raw = list(self.records)
            # 色をロック内でまとめて確定（再読み込みでも一貫, 課題②）
            key_colors = {str(r["speaker"]): self.html_color(r["speaker"])
                          for r in raw if "speaker" in r}
            records = []
            for r in raw:
                if "sys" in r:
                    records.append({"type": "sys", "ms": r.get("ms"),
                                    "text": r["sys"]})
                elif "speaker" in r:
                    records.append({
                        "type": "utt",
                        "ms": r.get("ms"), "end_ms": r.get("end_ms"),
                        "speaker": self.disp_name(r["speaker"]),
                        "color": key_colors[str(r["speaker"])],
                        "text": r.get("text", ""),
                        "corrected": r.get("vp") == "補正",
                        "bc": bool(r.get("bc")),
                        "unsure": str(r["speaker"]) == UNSURE_SPEAKER,
                        "speaker_source": r.get("speaker_source"),
                        "speaker_confidence": r.get("speaker_confidence"),
                        "speaker_reason": r.get("speaker_reason"),
                    })
            speakers = []
            _seen: set[str] = set()
            for r in raw:
                key = str(r.get("speaker", "")) if "speaker" in r else ""
                if not key or key in NON_PARTICIPANT_KEYS:
                    continue  # AI話者・未確定はリネーム対象外
                name = self.disp_name(key)
                if name in _seen:
                    continue
                _seen.add(name)
                label = self._speaker_label(key)
                speakers.append({"name": name, "label": label,
                                 "color": key_colors[key],
                                 "renameable": label is not None})
        with self.topics_lock:
            topics = []
            for t in self.topics:
                speaker = t.get("speaker", "")
                display_speaker = (
                    speaker
                    if speaker in self._AGENDA_SPEAKERS
                    else self.disp_name(speaker)
                )
                topics.append({"topic": t.get("topic", ""), "speaker": display_speaker})
        stats = participation_stats(
            raw, exclude_speakers=NON_PARTICIPANT_KEYS)
        participation = [
            {"speaker": self.disp_name(sp),
             "color": key_colors.get(str(sp), "#888"),
             "time_share": round(d["time_share"], 3),
             "char_share": round(d["char_share"], 3),
             "turn_share": round(d["turn_share"], 3),
             "turns": d["turns"], "chars": d["chars"],
             "has_time": d["talk_ms"] > 0}
            for sp, d in stats.items()
        ]
        agent = None
        if self.agent is not None:
            agent = {"enabled": self.agent.enabled, "mode": self.agent.mode,
                     "voice": self.agent.voice,
                     "model": getattr(self.agent, "model", None)}
        return {
            "rev": self.rev,
            "mode": self.session_mode(),
            "running": not self.stop.is_set(),
            "resetting": self.resetting,
            "setup": {"waiting": self.waiting_to_start},
            "vp": {"enabled": self.tracker is not None,
                   "model": getattr(self.tracker, "model", None),
                   "locked": self.tracker is not None and not self.tracker.auto,
                   "roster": (self.tracker.active_profile_names()
                              if self.tracker is not None else [])},
            "diarization": {
                "provider": getattr(self.diarization_provider, "name", None),
                "max_speakers": getattr(self.args, "diarization_max_speakers", None),
            },
            "stt": {
                "provider": getattr(self.stt_backend, "name", None),
                "model": getattr(self.args, "model", None),
                "lang": getattr(self.args, "lang", None),
            },
            "intervention": {
                "enabled": self.intervention_enabled,
                "proactivity": self.proactivity_name,
                "trigger_n": getattr(self.agent, "trigger_n", None),
                "model": getattr(self.agent, "model", None),
                # 手動呼び出しボタンの有効判定用（agent が有効=facilitate/converse）。
                "agent_active": self.agent is not None and self.agent.enabled,
                # 直近の手動呼び出しの進行状況（受付済み/待機中/発話済み/失敗）。
                "manual_call": self.manual_call_status,
            },
            "intervention_events": list(self.intervention_events[-20:]),
            "agenda": self._current_agenda(),
            "started": self.started.strftime("%Y-%m-%d %H:%M"),
            "partial": {"speaker": self.partial_speaker, "text": self.partial_text},
            "speakers": speakers,
            "records": records,
            "topics": topics,
            "participation": participation,
            "agent": agent,
        }

    # ------------------------------------------------------------------
    # 録音(WAV)の開始・確定
    # ------------------------------------------------------------------
    def open_wav(self):
        """録音wavを開きヘッダを書く。PCMバッファ関連もリセットする.

        STT接続を作り直す際、新しいSTTのタイムスタンプ(ms=0起点)と
        PCMバッファの位置を揃えるため、バッファのオフセット類も0に戻す。
        """
        self.pcm_buf = bytearray()
        self.pcm_buf_offset = 0
        self.pcm_total_bytes = 0
        self.asr_pcm_buf = bytearray()
        self.asr_pcm_buf_offset = 0
        self.asr_pcm_total_bytes = 0
        self.stt_time_offset_ms = 0
        self._stt_connection_audio_base_bytes = 0
        with self._ai_speech_lock:
            self._ai_speech_intervals.clear()
            self._ai_speech_open.clear()
            self.retired_echo_texts.clear()
        try:
            self.pcm_file = open(self.wav_path, "wb")  # noqa: SIM115
            self.pcm_file.write(b"RIFF" + struct.pack("<I", 0) + b"WAVEfmt " +
                                struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
                                b"data" + struct.pack("<I", 0))
            self.pcm_file.flush()
        except OSError as e:
            print(f"# 録音ファイルを開けません: {e}", flush=True)
            self.pcm_file = None

    def finalize_wav(self) -> str | None:
        """録音wavのヘッダを確定して閉じる。保存したパスを返す（短すぎ/失敗ならNone）.

        中身は「STTへ送れた音声」だけなので、長さは `asr_pcm_total_bytes`。
        取り込んだのに送れなかった分（接続断・送信失敗）は録音に入らない——
        入れると wav と発話 ms がずれ、後から採点にもアノテーションにも
        使えなくなるため。捨てた秒数はここで知らせる。
        """
        if self.pcm_file is None:
            return None
        dropped = max(self.pcm_total_bytes - self.asr_pcm_total_bytes, 0)
        if dropped > SR * 2:  # 1秒以上こぼれたときだけ知らせる
            print(f"# 録音: STTへ送れず録音に含めなかった音声 "
                  f"{dropped / (SR * 2):.1f}秒", flush=True)
        try:
            self.pcm_file.flush()
            data_size = self.asr_pcm_total_bytes
            self.pcm_file.seek(4)
            self.pcm_file.write(struct.pack("<I", 36 + data_size))
            self.pcm_file.seek(40)
            self.pcm_file.write(struct.pack("<I", data_size))
            self.pcm_file.close()
        except OSError:
            self.pcm_file = None
            return None
        self.pcm_file = None
        if data_size > SR * 2 * 10:
            return self.wav_path
        with contextlib.suppress(OSError):
            os.remove(self.wav_path)
        return None

    def mark_stt_connection_started(self) -> None:
        """新しいSTT接続の時刻0と、送信済み音声の絶対位置を対応させる."""
        with self.buf_lock:
            base = self.asr_pcm_total_bytes
            self._stt_connection_audio_base_bytes = base
            self.stt_time_offset_ms = int(base / (SR * 2) * 1000)

    def stt_abs_ms(self, ms: int | None) -> int | None:
        """現在のSTT接続内タイムスタンプを、会議全体の時刻に変換する."""
        if ms is None:
            return None
        return int(ms) + self.stt_time_offset_ms

    # ------------------------------------------------------------------
    # AI再生区間（エコー判定の区間ベース化, P2-1）
    # ------------------------------------------------------------------
    def current_asr_ms(self) -> int:
        """マイク音声の現在位置をmsで返す（cur_ms/cur_end と同じタイムライン）.

        16kHz・16bit なので 32 bytes/ms。int の読み取りは CPython では原子的で、
        エコーのマージン用途には十分な精度なのでロックは取らない。
        """
        return self.asr_pcm_total_bytes // 32

    def note_ai_speech_start(self, source: str) -> None:
        """AI（source: agent/partner）の再生開始を現在のマイクms位置で記録する."""
        with self._ai_speech_lock:
            self._ai_speech_open[source] = self.current_asr_ms()

    def note_ai_speech_end(self, source: str) -> None:
        """AI再生の終了で開いていた区間を閉じ、判定用の履歴に積む."""
        with self._ai_speech_lock:
            start = self._ai_speech_open.pop(source, None)
            if start is None:
                return
            end = max(self.current_asr_ms(), start)
            self._ai_speech_intervals.append((start, end, source))

    def add_retired_echo_texts(self, texts) -> None:
        """切断される発話元（partner等）の直近応答テキストをエコー参照に退避する（P2-4）."""
        now = time.monotonic()
        with self._ai_speech_lock:
            for t in texts:
                s = str(t).strip()
                if s:
                    self.retired_echo_texts.append((now, s))

    def recent_retired_echo_texts(self, *, now: float | None = None) -> list[str]:
        """TTL 内の退役エコーテキストを返す（テキスト安全網の照合対象用）."""
        cur = time.monotonic() if now is None else now
        with self._ai_speech_lock:
            return [t for (ts, t) in self.retired_echo_texts
                    if cur - ts < self._RETIRED_ECHO_TTL]

    def has_ai_speech_intervals(self) -> bool:
        """区間ベース判定に使える記録があるか（無ければ壁時計にフォールバック）."""
        with self._ai_speech_lock:
            return bool(self._ai_speech_intervals or self._ai_speech_open)

    def overlaps_ai_speech(self, start_ms: int | None, end_ms: int | None,
                           *, source: str | None = None,
                           margin_ms: int = 300) -> bool:
        """発話区間 [start_ms, end_ms] がAI再生区間と重なるか（±margin_ms）.

        閉じた区間に加え、まだ再生中の開いた区間（[start, 現在]）も対象にする。
        source を指定すると、その発話元（agent/partner）の区間だけを見る。
        """
        if start_ms is None or end_ms is None:
            return False
        lo, hi = start_ms - margin_ms, end_ms + margin_ms
        with self._ai_speech_lock:
            for a, b, src in self._ai_speech_intervals:
                if source is not None and src != source:
                    continue
                if a <= hi and lo <= b:
                    return True
            if self._ai_speech_open:
                now = self.current_asr_ms()
                for src, a in self._ai_speech_open.items():
                    if source is not None and src != source:
                        continue
                    if a <= hi and lo <= now:
                        return True
        return False

    def reset_for_new_meeting(self) -> dict:
        """現在の会議を確定保存し、同一プロセスのまま次の会議に切り替える（F6）.

        声紋プロファイル・話者名・色は引き継ぐ（同じメンバーの次の会議向け）。
        議事録・論点・各カーソル・キュー・エージェントの保留はクリアする。
        録音(wav)とSTT/エージェント接続は継続する。
        """
        self.save(live=False)  # 現在の会議を確定保存（ファイルは残る）
        self.finalize_wav()    # 現在の録音を確定（会議ごとにwavを分ける）

        # 新しい会議の出力先（既存と同じディレクトリ＋新タイムスタンプ）
        self.started = datetime.datetime.now()
        base_dir = os.path.dirname(self.out_path) or "transcripts"
        base = os.path.join(base_dir, self.started.strftime("%Y-%m-%d_%H%M%S"))
        self.out_path = base + ".md"
        self.html_path = base + ".html"
        self.diag_path = base + ".diag.jsonl"
        self.turns_path = base + ".turns.jsonl"
        self.interventions_path = self._interventions_path_for(self.turns_path)
        self.intervention_review_path = self._intervention_review_path_for(self.turns_path)
        self.wav_path = base + ".wav"
        self.open_wav()        # 新しい録音を開く（PCMバッファもリセットしSTTのmsと整合）

        # 状態クリア（課題③: 話者ラベリングもリセット。永続化は別機能）
        with self.state_lock:
            # 世代を進める（H2）。以降 worker がこの epoch 不一致を見て、リセットを
            # 跨いだ古い計算結果の書き戻しを破棄する。records クリアと同一 lock 内で
            # 行うことで、worker から見て「records が空 ⇔ epoch が新しい」を一貫させる。
            self.meeting_epoch += 1
            self.records = []
            self.names = {}
            self.colors = {}
            self.html_color_idx = {}
            self.html_color_seq = 0
            self.anonymous_labels = {}
            self.agent_cursor = 0
            self.partial_text = ""
            self.partial_speaker = ""
            self.diarization_speaker_keys = {}
            self.diarization_pending_ms = {}
            self.diarization_key_seq = 0
            self.constrain_drop_counts = {}
            self.constrain_warned = False
        with self.diarization_lock:
            self.diarization_events = []
        if self.cluster_namer is not None:
            self.cluster_namer.reset()
        if self.seat_audio is not None:
            self.seat_audio.reset()
        if self.retro is not None:
            self.retro.reset()
        if self.tracker is not None:
            self.tracker.reset_session()
        with self.topics_lock:
            self.topics = []
        self.topic_cursor = 0
        self.drift_cursor = 0
        self.fact_cursor = 0
        self.triage_cursor = 0
        self.intervention_events = []
        self._intervention_event_seq = 0
        self._last_intervention_event_id = None
        self._last_intervention_event_reason = None
        self.manual_call_status = None
        self._last_utt_time[0] = time.monotonic()
        self._was_in_echo[0] = False
        for q in (self.drift_requests, self.invite_requests, self.factcheck_requests,
                  self.manual_call_requests, self.summarize_requests):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.agent is not None:
            self.agent.reset_meeting()
        should_wait_for_setup = bool(
            getattr(self.args, "setup", True)
            and self._serve
            and not getattr(self.args, "wav", None)
            and not getattr(self.args, "simulate", None)
        )
        self.waiting_to_start = should_wait_for_setup
        if should_wait_for_setup:
            self.start_requested.clear()
        self.rev += 1
        self.save()  # 空の新会議ファイルを作成
        return {"ok": True, "started": self.started.strftime("%Y-%m-%d %H:%M:%S"),
                "waiting": self.waiting_to_start}

    _AGENDA_SPEAKERS = ("議題", "議題(自動)")

    def set_agenda(self, topic: str) -> dict:
        """UIから議題（脱線判定の基準）を設定/変更する.

        既存の議題エントリを差し替える（抽出済みの論点は残す）。
        空文字なら議題を削除する。次回以降の脱線判定に即反映される。
        """
        topic = (topic or "").strip()
        with self.topics_lock:
            self.topics = [t for t in self.topics
                           if t.get("speaker") not in self._AGENDA_SPEAKERS]
            if topic:
                self.topics.insert(0, {"topic": topic, "speaker": "議題"})
        self.rev += 1
        self.save()
        return {"ok": True, "agenda": topic}

    def add_intervention_event(self, reason: str, detail: str = "",
                               metadata: dict | None = None) -> None:
        """UIで確認するための介入理由ログを追加する."""
        now = datetime.datetime.now()
        self._intervention_event_seq += 1
        event_id = f"int-{self._intervention_event_seq:04d}"
        self._last_intervention_event_id = event_id
        self._last_intervention_event_reason = reason
        event = {
            "time": now.strftime("%H:%M:%S"),
            "reason": reason,
            "detail": detail,
        }
        self.intervention_events.append(event)
        del self.intervention_events[:-20]
        self.write_intervention_event({
            "event_id": event_id,
            "type": "trigger",
            **event,
            "created_at": now.isoformat(timespec="seconds"),
            "meeting_started": self.started.isoformat(timespec="seconds"),
            "metadata": metadata or {},
        })
        self.rev += 1

    def set_proactivity(self, name: str) -> dict:
        """UIから介入頻度プロファイルを更新する."""
        if name not in _PROACTIVITY_PROFILES:
            return {"ok": False, "error": "介入頻度は controlled / standard / active で指定してください"}
        self.proactivity_name = name
        self.proactivity = dict(_PROACTIVITY_PROFILES[name])
        try:
            self.args.proactivity = name
        except AttributeError:
            self.args = SimpleNamespace(proactivity=name)
        self.rev += 1
        return {"ok": True, "proactivity": name}

    def set_intervention_enabled(self, enabled: bool) -> dict:
        """UIからファシリテーター介入の有効/無効を切り替える."""
        self.intervention_enabled = bool(enabled)
        if not self.intervention_enabled and self.agent is not None:
            with contextlib.suppress(Exception):
                self.agent.interrupt()
            with contextlib.suppress(Exception):
                self.agent.reset_meeting()
        self.rev += 1
        return {"ok": True, "enabled": self.intervention_enabled}

    def queue_manual_facilitator_call(self, request: str = "",
                                      source: str = "ui") -> dict:
        """参加者からの手動呼び出しをキューに積む（Phase1）.

        直接 agent.trigger() はしない。既存の _run_agent_worker + Controller 経路で
        他の候補と同じく採否される。実際に採択された時に介入ログへ残す。
        """
        if not self.intervention_enabled:
            return {"ok": False, "error": "介入がオフのため呼び出せません"}
        agent = self.agent
        if agent is None or getattr(agent, "mode", "off") == "off":
            return {"ok": False, "error": "ファシリテーターが無効です"}
        # 改行を空白へ正規化し、長すぎる依頼は切り詰める。
        text = re.sub(r"\s+", " ", str(request or "")).strip()
        if len(text) > _MANUAL_CALL_MAX_CHARS:
            text = text[:_MANUAL_CALL_MAX_CHARS]
        self.manual_call_requests.put({
            "request": text,
            "source": source,
            "created_at": time.monotonic(),
            "created_wall_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        # UIステータス: 受付済み（連打時は最新の依頼で上書き）。
        self.set_manual_call_status("queued", source=source, request=text)
        return {"ok": True, "queued": True, "request": text}

    def set_manual_call_status(self, status: str, *, detail: str = "",
                               source: str | None = None,
                               request: str | None = None,
                               wait_sec: float | None = None) -> None:
        """手動呼び出しの進行状況（受付済み/待機中/発話済み/失敗）を更新する.

        UI表示・観測用の軽量ステータス。ワーカーの0.25秒ループから呼ばれても
        SSEが洪水しないよう、表示内容が変わる時だけ rev を上げる。
        """
        prev = self.manual_call_status or {}
        wait_rounded = round(wait_sec, 0) if wait_sec is not None else None
        next_source = source if source is not None else prev.get("source")
        next_request = request if request is not None else prev.get("request")
        if (prev.get("status") == status and prev.get("detail") == detail
                and prev.get("wait_sec") == wait_rounded
                and prev.get("source") == next_source
                and prev.get("request") == next_request):
            return
        self.manual_call_status = {
            "status": status,
            "detail": detail,
            "source": next_source,
            "request": next_request,
            "at": datetime.datetime.now().strftime("%H:%M:%S"),
            "wait_sec": wait_rounded,
        }
        self.rev += 1

    def set_diarization_max_speakers(self, value: int | None) -> dict:
        """UIから想定話者数ヒントを更新する.

        STT/外部diarizationの多くは接続開始時にしかmax_speakersを受け取れないため、
        バックエンドへの反映は次の会議リセット時の再接続になる。ただし
        ``constrain_human_speaker_key`` と ``tracker.set_max_human_speakers`` は
        ``args.diarization_max_speakers`` を直接参照するため**即時に**厳格化/緩和
        される。会議中に減らしても既存の席（表示済み参加者）は維持され、
        以後の新規参入だけが絞られる（統一席ルール, 2026-07-25。旧仕様の
        「文字辞書順で既存を未確定化」は 2026-07-14 型事故の機構のため廃止）。
        """
        if value is not None and not 1 <= value <= 10:
            return {"ok": False, "error": "話者数は1〜10で指定してください"}
        try:
            self.args.diarization_max_speakers = value
        except AttributeError:
            self.args = SimpleNamespace(diarization_max_speakers=value)
        backend = getattr(self, "stt_backend", None)
        if backend is not None and hasattr(backend, "set_max_speakers"):
            backend.set_max_speakers(value)
        provider = self.diarization_provider
        if provider is not None and hasattr(provider, "set_max_speakers"):
            provider.set_max_speakers(value)
        if self.tracker is not None and hasattr(self.tracker, "set_max_human_speakers"):
            self.tracker.set_max_human_speakers(value)
        self.rev += 1
        return {"ok": True, "max_speakers": value}

    def _current_agenda(self) -> str:
        with self.topics_lock:
            for t in self.topics:
                if t.get("speaker") in self._AGENDA_SPEAKERS:
                    return t.get("topic", "")
        return ""

    def seed_topic(self, topic: str | None, speaker: str = "議題"):
        """明示的な議題を脱線検出の基準論点としてシードする（Fix 8）.

        debate/simulate モードのように議題が分かっている場合、論点抽出LLMの
        成否を待たずに最初から脱線検出を効かせるための初期基準を入れる。
        既に論点があれば何もしない（抽出済みを優先）。
        """
        if not topic:
            return
        with self.topics_lock:
            if not self.topics:
                self.topics.append({"topic": topic, "speaker": speaker})

    def key_of(self, tok: str) -> str:
        """コマンド引数を表示キーへ: 人物名はそのまま、数字はそのラベルの現在の表示先."""
        if self.tracker is not None:
            if tok in self.tracker.profiles:
                return tok
            if tok in self.tracker.sp_map:
                return self.tracker.sp_map[tok]
        return "#" + tok

    def show_partial(self, sp, text: str):
        # UI向けに途中経過(partial)を保持（認識中であることが分かるように, 課題①）
        t = text.strip()
        # peek_disp_name（割当てなし）を使う: partial 限りの幻キーにラベル文字を
        # 割り当てない。本表示（flush→records）の disp_name は従来どおり割当てあり。
        # 表示名の解決はロックの外で済ませる（端末出力と同じく重い処理を
        # ロック内に持ち込まない）。
        speaker = self.peek_disp_name(self.key_for_label(sp)) if t else ""
        # M6: 長い発話の途中では STT 確定レコードが来ず「喋っている最中に沈黙が
        # 伸びる」ため、pause 判定を満たした介入が発話に被さり得る。partial の受信
        # でも沈黙タイマーを更新し、発話中を「沈黙」と誤認しないようにする。
        # 「変化した場合のみ」更新するのは、同一 partial の再送で沈黙が永久に 0 に
        # 張り付くのを防ぐため。
        # 既知のトレードオフ: エコーウィンドウ中の AI 自身の声の partial でもタイマー
        # が更新され得るが、フロア判定を保守側（介入を待つ側）に倒すだけなので許容。
        #
        # partial_text と _last_partial_change は **組で** 読まれる（_workers の
        # _effective_silence が「非空の partial が直近に更新されていればフロアは
        # 埋まっている」と判定する, F3）。書き手側がロックを取らないと、読み手が
        # ロックを取っていても「新しいテキスト＋古いタイムスタンプ」の組を読みうる。
        # その場合フロアが空いたと誤認して介入が発話に被さる（M6/F3 が塞いだはずの
        # 穴が書き手側で開いていた。2026-07-25 監査）。ここで組にして書く。
        with self.state_lock:
            prev = self.partial_text
            self.partial_text = t
            self.partial_speaker = speaker
            if t and t != prev:
                _now = time.monotonic()
                self._last_utt_time[0] = _now
                self._last_partial_change = _now   # F3: フロア占有の鮮度判定に使う
        if not t:
            sys.stdout.write(CLEAR_LINE)
        else:
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            line = f"{self.partial_speaker}: {t}"
            sys.stdout.write(CLEAR_LINE + DIM + line[-(cols - 2):] + RESET)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 出力
    # ------------------------------------------------------------------
    def write_md(self, recs=None, path=None):
        with self.state_lock:
            rs = self.records if recs is None else recs
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            lines = [
                f"# 議事録 {self.started.strftime('%Y-%m-%d %H:%M')}",
                "",
                "話者: " + (", ".join(self.disp_name(s) for s in speakers) or "（未検出）"),
                "",
            ]
            for r in rs:
                if "sys" in r:
                    lines.append(f"> [{fmt_ts(r['ms'])}] {r['sys']}")
                    continue
                mark = " ⚡" if r.get("vp") == "補正" else ""
                lines.append(f"- **[{fmt_ts(r['ms'])}] {self.disp_name(r['speaker'])}{mark}**: {r['text']}")
            dst = path or self.out_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, dst)

    def write_html(self, live: bool = True, recs=None, path=None, status=None):
        import html as _html
        with self.state_lock:
            rs = self.records if recs is None else recs
            parts = []
            for r in rs:
                if "sys" in r:
                    parts.append(f'<div class="sys">⚙ {_html.escape(r["sys"])}</div>')
                    continue
                sp = str(r["speaker"])
                # 監査E: 旧実装 list(colors).index() は rekey の pop で後続全員の
                # 色がずれる（api_snapshot 側は C11/P2 で修正済みだった）。
                # 保存用HTMLも同じ安定採番 html_color() に統一する。
                c = self.html_color(sp)
                badge = ""
                if r.get("vp") == "補正":
                    note = _html.escape(r.get("note", ""))
                    badge = f'<span class="badge" title="{note}">⚡声紋補正</span>'
                parts.append(
                    f'<div class="u"><span class="ts">{fmt_ts(r["ms"])}</span>'
                    f'<span class="who" style="color:{c}">{_html.escape(self.disp_name(sp))}</span>'
                    f'{_html.escape(r["text"])}{badge}</div>'
                )
            speakers = list(dict.fromkeys(r["speaker"] for r in rs if "speaker" in r))
            sp_tags = []
            for s in speakers:
                dn = _html.escape(self.disp_name(s))
                c = self.html_color(s)
                lbl = self._speaker_label(str(s))
                is_renameable = self._serve and self.tracker is not None and lbl is not None
                if is_renameable:
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'<div class="rename-row">'
                        f'<input class="rename-input" placeholder="名前" data-label="{_html.escape(lbl)}">'
                        f'<button class="rename-btn" onclick="rename(this)">登録</button>'
                        f'</div></div>')
                else:
                    sp_tags.append(
                        f'<div class="speaker-tag">'
                        f'<div class="speaker-name"><span class="dot" style="background:{c}"></span>{dn}</div>'
                        f'</div>')
            if sp_tags:
                speaker_panel = ('<div class="sidebar"><p class="sidebar-title">この会議の話者</p>'
                                 '<div class="speaker-panel">' + ''.join(sp_tags) + '</div></div>')
            else:
                speaker_panel = ''
            profile_panel = ''
            if self._serve and self.tracker is not None:
                all_names = self.tracker.all_profile_names()
                if all_names:
                    active_names = set(self.tracker.active_profile_names())
                    items = []
                    for n in all_names:
                        cls = 'profile-item active' if n in active_names else 'profile-item'
                        items.append(
                            f'<div class="{cls}" data-name="{_html.escape(n)}">'
                            f'<span class="profile-toggle" '
                            f'onclick="toggleProfile(this.parentElement)"></span>'
                            f'<span class="profile-name" '
                            f'onclick="toggleProfile(this.parentElement)">'
                            f'{_html.escape(n)}</span>'
                            f'<span class="profile-del" title="この声の登録を削除"'
                            f' onclick="forgetProfile(this.parentElement)">×</span>'
                            f'</div>')
                    profile_panel = ('<div class="profile-section">'
                                     '<p class="sidebar-title">プロファイル'
                                     '<span class="sidebar-hint">'
                                     'クリックで有効/無効・×で削除</span></p>'
                                     + ''.join(items) + '</div>')
            stats_panel = ''
            talk_rs = [r for r in rs if "speaker" in r and r.get("text")]
            if talk_rs:
                sp_dur: dict[str, float] = {}
                sp_chars: dict[str, int] = {}
                sp_turns: dict[str, int] = {}
                for r in talk_rs:
                    s = r["speaker"]
                    ms, end = r.get("ms"), r.get("end_ms")
                    dur = (end - ms) / 1000.0 if ms is not None and end is not None and end > ms else 0.0
                    sp_dur[s] = sp_dur.get(s, 0.0) + dur
                    sp_chars[s] = sp_chars.get(s, 0) + len(r["text"])
                    sp_turns[s] = sp_turns.get(s, 0) + 1
                total_dur = sum(sp_dur.values()) or 1.0
                total_chars = sum(sp_chars.values()) or 1
                total_turns = sum(sp_turns.values()) or 1
                ranked = sorted(sp_dur.keys(), key=lambda s: sp_dur[s], reverse=True)

                def _bar_rows(data, total, unit=""):
                    rows = []
                    for s in ranked:
                        v = data.get(s, 0)
                        pct = v / total * 100 if total else 0
                        c = self.html_color(s)
                        dn = _html.escape(self.disp_name(s))
                        short = dn[:2] if len(dn) > 3 else dn
                        rows.append(
                            f'<div class="stats-row">'
                            f'<span class="stats-name" title="{dn}">{short}</span>'
                            f'<div class="stats-bar-bg">'
                            f'<div class="stats-bar" style="width:{pct:.0f}%;background:{c}"></div>'
                            f'</div>'
                            f'<span class="stats-pct">{pct:.0f}%</span>'
                            f'</div>')
                    return ''.join(rows)

                groups = []
                if total_dur > 0.5:
                    groups.append('<div class="stats-group">'
                                  '<div class="stats-label">発話時間</div>'
                                  + _bar_rows(sp_dur, total_dur) + '</div>')
                groups.append('<div class="stats-group">'
                              '<div class="stats-label">文字数</div>'
                              + _bar_rows(sp_chars, total_chars) + '</div>')
                groups.append('<div class="stats-group">'
                              '<div class="stats-label">発話回数</div>'
                              + _bar_rows(sp_turns, total_turns) + '</div>')
                stats_panel = ('<div class="stats-section">'
                               '<p class="sidebar-title">発言量</p>'
                               + ''.join(groups) + '</div>')
            topics_panel = ''
            with self.topics_lock:
                if self.topics:
                    items = []
                    for t in self.topics:
                        tt = _html.escape(t.get("topic", ""))
                        ts = _html.escape(t.get("speaker", ""))
                        items.append(f'<div class="topic-item">'
                                     f'<div class="topic-text">{tt}</div>'
                                     f'<div class="topic-by">{ts}</div></div>')
                    topics_panel = ('<div class="topics-section">'
                                   '<p class="sidebar-title">論点</p>'
                                   + ''.join(items) + '</div>')
            agent_panel = ''
            if self.agent is not None:
                cur_mode = self.agent.mode
                if self.agent._connected:
                    conn = '接続中'
                elif self.agent._conn_error:
                    conn = f'エラー: {_html.escape(self.agent._conn_error)}'
                else:
                    conn = '未接続'
                mode_btns = []
                for m, lbl in [("off", "OFF"), ("facilitator", "進行役"),
                               ("conversation", "会話")]:
                    cls = "agent-mode-btn active" if m == cur_mode else "agent-mode-btn"
                    mode_btns.append(f'<button class="{cls}" data-mode="{m}" '
                                     f'onclick="setAgentMode(this)">{lbl}</button>')
                voice_opts = []
                for v in AGENT_VOICES:
                    sel = 'selected' if v == self.agent.voice else ''
                    voice_opts.append(f'<option value="{v}" {sel}>{v}</option>')
                trigger_val = self.agent.trigger_n
                agent_panel = (
                    f'<div class="agent-section" data-mode="{cur_mode}">'
                    f'<div class="agent-header">'
                    f'<span class="agent-label">🤖 AI Agent</span>'
                    f'<span class="agent-conn">{conn}</span>'
                    f'</div>'
                    f'<div class="agent-modes">{"".join(mode_btns)}</div>'
                    f'<div class="agent-opts">'
                    f'<label class="agent-opt-label">声'
                    f'<select class="agent-select" onchange="setAgentVoice(this)">'
                    f'{"".join(voice_opts)}</select></label>'
                    f'<label class="agent-opt-label agent-trigger-row">'
                    f'間隔 <input type="number" class="agent-num" value="{trigger_val}" '
                    f'min="1" max="50" onchange="setAgentTrigger(this)">発話'
                    f'</label>'
                    f'</div></div>')
            doc = HTML_TMPL.format(
                refresh='<meta http-equiv="refresh" content="2">' if live else "",
                title=self.started.strftime("%Y-%m-%d %H:%M"),
                status=status or ('<span class="live">● ライブ（2秒ごと自動更新）</span>'
                                  if live else "終了"),
                speaker_panel=speaker_panel,
                profile_panel=profile_panel,
                stats_panel=stats_panel,
                topics_panel=topics_panel,
                agent_panel=agent_panel,
                body="\n".join(parts) or '<p class="meta">（まだ発話なし）</p>',
            )
            dst = path or self.html_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(doc)
            os.replace(tmp, dst)

    def write_turns(self, recs=None, path=None):
        """discussion-support(das)のUtteranceスキーマでJSONL出力."""
        with self.state_lock:
            rs = self.records if recs is None else recs
            lines = []
            tid = 0
            for r in rs:
                if "speaker" not in r or not r.get("text"):
                    continue
                tid += 1
                lines.append(json.dumps({"turn_id": tid, "speaker": self.disp_name(r["speaker"]),
                                         "text": r["text"], "ms": r.get("ms"),
                                         "end_ms": r.get("end_ms")},
                                        ensure_ascii=False))
            dst = path or self.turns_path
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, dst)

    def write_intervention_event(self, event: dict, path=None):
        """介入イベントを追記保存する."""
        dst = path or self.interventions_path
        with open(dst, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_intervention_review(self, entry: dict, path=None):
        """採否レビューを追記保存する.

        従来の介入ログ（``.interventions.jsonl``）とは別ファイルに分け、
        「介入候補・採択した Controller 判断・抑制理由・latency」を後から
        追えるようにする（なぜ話したか／黙ったかの観測用）。
        """
        dst = path or self.intervention_review_path
        with open(dst, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add_intervention_review(self, entry: dict) -> None:
        """Controller の採否判断1件を ``intervention_review.jsonl`` に記録する.

        注: ``type`` は歴史的に ``shadow_decision``（Phase1 で shadow 判断を
        ログしていた頃の名残）。既存ログとの互換のため値は据え置く。
        """
        now = datetime.datetime.now()
        payload = {
            "type": "shadow_decision",
            "time": now.strftime("%H:%M:%S"),
            "created_at": now.isoformat(timespec="seconds"),
            "meeting_started": self.started.isoformat(timespec="seconds"),
            **entry,
        }
        self.write_intervention_review(payload)

    def add_facilitator_delivery_event(self, text: str,
                                       timing: dict | None = None) -> None:
        """実際に参加者へ届いたファシリテーター発話を介入ログへ残す.

        timing: 任意の観測情報（例: speak_start_latency_ms = trigger→発話開始の遅延）。
        """
        text = text.strip()
        if not text or "介入不要" in text:
            return
        now = datetime.datetime.now()
        event = {
            "type": "delivery",
            "trigger_event_id": self._last_intervention_event_id,
            "time": now.strftime("%H:%M:%S"),
            "created_at": now.isoformat(timespec="seconds"),
            "meeting_started": self.started.isoformat(timespec="seconds"),
            "speaker": "ファシリテーター",
            "text": text,
        }
        if timing:
            event["timing"] = timing
        self.write_intervention_event(event)
        # 手動呼び出しの発話が実際に届いたら「発話済み」へ（UX観測）。
        if (self._last_intervention_event_reason == "manual_call"
                and (self.manual_call_status or {}).get("status") == "dispatched"):
            self.set_manual_call_status("delivered")

    def save(self, live: bool = True):
        self.rev += 1  # 変更を通知（SSEの差分配信用, F2）
        self.write_md()
        if not self._serve:
            self.write_html(live)
        self.write_turns()
