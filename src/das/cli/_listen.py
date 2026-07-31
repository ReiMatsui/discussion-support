"""listen / listen-soniox サブコマンド。"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from das.cli import app
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.runtime import Orchestrator
from das.settings import get_settings
from das.types import Utterance


@app.command(name="listen")
def listen(
    docs: Path | None = typer.Option(
        None,
        "--docs",
        help="議論前に取り込む文書ディレクトリ (省略時は data/docs)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="出力先サブディレクトリ名 (省略時は ISO タイムスタンプ)",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="リンク採用の信頼度閾値 (省略時は設定値)",
    ),
    top_k: int = typer.Option(5, "--top-k", help="リンク候補の embedding top-k (混合)"),
    top_k_per_source: int = typer.Option(
        0,
        "--top-k-per-source",
        help="source 別 top-k モード。0 (= 既定) なら混合 top-k を使う。"
        "正の値で utterance / document / web の各バケットから top-N を取る (Fix A)",
    ),
    skip_docs: bool = typer.Option(
        False, "--skip-docs", help="ドキュメントの事前 AF 化をスキップ"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="ASR モデル名 (省略時は DAS_ASR_MODEL = large-v3)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="ASR バックエンド (省略時は DAS_ASR_BACKEND = mlx-whisper)",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="認識言語 (省略時は DAS_ASR_LANGUAGE = ja)",
    ),
) -> None:
    """マイクからの音声をリアルタイム文字起こしして AF を構築する (asr extras 必須)。"""

    asyncio.run(
        _run_listen_async(
            docs=docs,
            run_id=run_id,
            threshold=threshold,
            top_k=top_k,
            top_k_per_source=top_k_per_source if top_k_per_source > 0 else None,
            skip_docs=skip_docs,
            model=model,
            backend=backend,
            language=language,
        )
    )


@app.command(name="listen-soniox")
def listen_soniox(
    docs: Path | None = typer.Option(
        None, "--docs", help="議論前に取り込む文書ディレクトリ (省略時は data/docs)"
    ),
    skip_docs: bool = typer.Option(False, "--skip-docs", help="ドキュメントの事前 AF 化をスキップ"),
    wav: Path | None = typer.Option(
        None,
        "--wav",
        help="実マイクの代わりに WAV ファイルで擬似ライブ入力する",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="想定話者数のヒント (文字起こし側の --diarization-max-speakers に転送)",
    ),
    soniox_args: str = typer.Option(
        "",
        "--soniox-args",
        help="文字起こし側へ渡す追加引数 (空白区切り。例: '--stt speechmatics')。"
        "上記の第一級オプションと重複した場合はこちらが優先される",
    ),
) -> None:
    """Soniox+声紋プロファイルで「誰が何を」をライブ取得し、統合 AF 構築＋ライブ介入を行う。

    既定で推奨構成（Soniox + pyannote + 声紋のハイブリッド話者帰属）で動く。

    使用例:
      das listen-soniox                      (マイクから開始)
      das listen-soniox --wav meeting.wav   (録音ファイルで擬似ライブ)
      das listen-soniox --soniox-args "--diarization none"  (Soniox単独に落とす)

    speaker-attribution 由来の話者特定つき文字起こし (das.asr.live) を
    別スレッドで走らせ、確定発話を Orchestrator.run_live に流す。
    FacilitationAgent が周期的に介入を判定し、ターミナルとライブ議事録 HTML
    (💡システム行) に提示する。
    要: SONIOX_API_KEY (.env) / `uv sync --extra soniox`。
    話者の実名登録は文字起こし側の標準入力で「1=名前」。
    """
    asyncio.run(
        _run_listen_soniox_async(
            docs=docs,
            run_id=None,
            threshold=None,          # リンク閾値は既定値（調整オプションは削除済み）
            top_k=5,
            top_k_per_source=None,
            skip_docs=skip_docs,
            soniox_argv=_build_soniox_argv(
                wav=wav,
                max_speakers=max_speakers,
                # --docs を明示したときだけ文字起こし側の AF ランタイムにも渡す
                # （AF 有効時のみ使われる。既定 data/docs を暗黙に二重取り込み
                #  させないため、明示指定に限る）
                af_docs=docs if not skip_docs else None,
                soniox_args=soniox_args,
            ),
            min_utt_chars=7,          # 相槌をAFへ流さない既定
            facilitate_interval=3.0,  # 介入判定の周期（秒）
        )
    )


# ライブ介入(グラフレーン)の同一内容再提示の抑止窓（秒）。
# _facilitation.py の _INTERVENTION_CONTENT_DEDUP_SEC（Realtime レーン）と
# 同じ思想・同じ 10 分。会話が同じ状態に留まる間の再提示を止め、
# 窓を過ぎたら（本当に再度価値があれば）再提示を許す。
_PRESENT_DEDUP_SEC = 600.0


def _is_duplicate_presentation(body: str, recent: dict[str, float],
                               *, now: float,
                               window: float = _PRESENT_DEDUP_SEC) -> bool:
    """同一内容（空白無視の完全一致）の介入提示を window 秒抑止する.

    True を返した場合は提示しない。False の場合は提示扱いとして
    recent に時刻を記録する（呼び出し側の分岐を単純に保つため副作用込み）。
    """
    key = re.sub(r"\s+", "", body or "")
    if not key:
        return False
    last = recent.get(key)
    if last is not None and now - last < window:
        return True
    recent[key] = now
    return False


def _build_soniox_argv(
    *,
    wav: Path | None = None,
    max_speakers: int | None = None,
    af_docs: Path | None = None,
    soniox_args: str = "",
) -> list[str]:
    """第一級オプションを文字起こし側 (das.asr.live, click) の argv に合成する。

    推奨構成（pyannote＋クラスタ名前付け）は文字起こし側の**既定**なので
    ここでは何も足さない。click は同一オプションが複数回渡されると最後の
    値を採用する (後勝ち)。``--soniox-args`` を末尾に置くことで、第一級
    オプションと重複した場合は ``--soniox-args`` 側が優先される。
    ``soniox_args`` は shlex で分割するため、クォートすれば空白を含む
    パス等も安全に渡せる。
    """
    argv: list[str] = []
    if wav is not None:
        argv += ["--wav", str(wav)]
    if af_docs is not None:
        argv += ["--docs", str(af_docs)]
    if max_speakers is not None:
        argv += ["--diarization-max-speakers", str(max_speakers)]
    if soniox_args:
        argv += shlex.split(soniox_args)
    return argv


# --- 実体 --------------------------------------------------------------


async def _run_listen_soniox_async(
    *,
    docs: Path | None,
    run_id: str | None,
    threshold: float | None,
    top_k: int,
    top_k_per_source: int | None,
    skip_docs: bool,
    soniox_argv: list[str],
    min_utt_chars: int = 7,
    facilitate_interval: float = 3.0,
) -> None:
    """live を別スレッドで回し、確定発話キュー → run_live ＋ 周期介入判定."""

    import threading
    from collections.abc import AsyncIterator

    from das.agents.facilitation import FacilitationAgent, InterventionDecision

    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[listen-soniox] run_dir={run_dir}")

    try:
        from das.asr import live as _live_mod
    except ImportError as exc:
        typer.echo(f"[listen-soniox] 依存が未インストールです: {exc}")
        typer.echo("`uv sync --extra soniox` を実行してください。")
        raise typer.Exit(1) from exc

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=run_dir / "graph.sqlite")
    orch = Orchestrator.assemble(
        llm=llm,
        store=store,
        threshold=threshold,
        top_k=top_k,
        top_k_per_source=top_k_per_source,
    )

    if not skip_docs and docs_dir.exists():
        typer.echo("[listen-soniox] ingesting documents...")
        await orch.ingest_documents(docs_dir)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _on_utt(speaker: str, text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (speaker, text))

    _live_mod.ON_UTTERANCE = _on_utt
    argv = ["--no-open", *soniox_argv]

    def _runner() -> None:
        try:
            _live_mod.main(argv)
        except BaseException as exc:
            typer.echo(f"\n[listen-soniox] 文字起こしスレッド終了: {exc!r}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_runner, daemon=True).start()

    history: list[Utterance] = []
    n_filtered = 0

    async def _utt_stream() -> AsyncIterator[Utterance]:
        nonlocal n_filtered
        turn = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            speaker, text = item
            turn += 1
            utt = Utterance(turn_id=turn, speaker=speaker, text=text)
            if len(text.strip()) < min_utt_chars:   # 相槌等はAFに流さない(コスト/ノイズ削減)
                n_filtered += 1
                continue
            history.append(utt)
            yield utt

    # --- ライブ介入: FacilitationAgent を周期的に呼び、ターミナル+議事録HTMLに提示 ---
    facilitator = FacilitationAgent(llm=llm)
    # 同一介入の再提示抑止（2026-07-25 実利用: 会話が停滞すると 3 秒周期の判定が
    # 同じ提示を延々と繰り返した）。このレーンには従来クールダウンも重複抑止も
    # 無かった。L1 は提示済み source_text を価値ゲート（B1 の新規性判定）に渡して
    # 再生成自体を抑え、最終防衛として本文の同一内容を 10 分窓で抑止する。
    presented_texts: set[str] = set()
    _recent_bodies: dict[str, float] = {}

    def _present(decision: InterventionDecision) -> None:
        if decision.kind == "skip":
            return
        if decision.kind == "l2":
            body = decision.brief or decision.reason
            head = "💡介入(全体)"
        else:
            to = decision.addressed_to or "発言者"
            parts = []
            for it in decision.items:
                tag = "支持" if it.relation == "support" else "反論"
                parts.append(f"[{tag}] {it.source_text}")
            body = " / ".join(parts) or decision.brief or decision.reason
            head = f"💡介入({to}さん宛)"
        if _is_duplicate_presentation(body, _recent_bodies,
                                      now=time.monotonic()):
            return
        for it in decision.items:
            presented_texts.add(it.source_text.strip())
        msg = f"{head}: {body}"
        typer.echo(f"\n{msg}")
        with contextlib.suppress(Exception):
            from das.asr import live as _live

            _live.post_system(msg)   # ライブ議事録(2秒自動更新HTML)に表示

    async def _facilitate_loop() -> None:
        while True:
            await asyncio.sleep(facilitate_interval)
            if not history:
                continue
            try:
                # 判断 + L2 整文を facilitation 側で一元化 (レビュー M-1)
                decision = await facilitator.decide_and_render(list(history), store)
                # L1 価値ゲート（B1）: AF checker レーンと同じ入口で「提示済み・
                # 直近既出・鮮度切れ」を落とす。従来この CLI レーンだけ未適用で、
                # 同一 L1 が無限に再提示されていた。
                decision = facilitator.apply_l1_value_gate(
                    decision, store, list(history),
                    presented_source_texts=presented_texts)
                _present(decision)
            except Exception as exc:
                typer.echo(f"[listen-soniox] 介入判定エラー: {exc!r}")

    fac_task = (
        asyncio.create_task(_facilitate_loop()) if facilitate_interval > 0 else None
    )

    typer.echo("[listen-soniox] 録音開始。実名登録は「1=名前」と入力。Ctrl-C で停止。")
    try:
        await orch.run_live(_utt_stream())
    except KeyboardInterrupt:
        typer.echo("\n[listen-soniox] 停止。スナップショットを保存します...")
    finally:
        if fac_task is not None:
            fac_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await fac_task
        if n_filtered:
            typer.echo(f"[listen-soniox] 相槌等のフィルタ: {n_filtered}発話をAF構築から除外")

    from das.viz import dump_snapshot

    snapshot_path = dump_snapshot(store, run_dir / "snapshot.json")
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))
    html_path: Path | None = None
    try:
        from das.viz import render_html

        html_path = render_html(store, run_dir / "graph.html")
    except ImportError as exc:
        typer.echo(f"[listen-soniox] HTML 生成をスキップ (viz extras 未導入: {exc})")
    summary = (
        f"\n[listen-soniox] done. nodes={n_nodes} edges={n_edges}\n"
        f"  snapshot -> {snapshot_path}"
    )
    if html_path is not None:
        summary += f"\n  html     -> {html_path}"
    typer.echo(summary)


async def _run_listen_async(
    *,
    docs: Path | None,
    run_id: str | None,
    threshold: float | None,
    top_k: int,
    top_k_per_source: int | None,
    skip_docs: bool,
    model: str | None,
    backend: str | None,
    language: str | None,
) -> None:
    """マイク → ASR → Orchestrator.run_live → snapshot 保存。

    ``run-session`` のライブ版。設計のキモは:
      - ASR 関連の重い import は **この関数の中まで遅延** させて、[asr] extras
        が無い環境でも CLI 自体は import できるようにしておく
      - マイク取り込みと utterance 消費を別タスクで走らせ、Ctrl-C 時には
        EOS (空フレーム) を ASR に送って残りの確定行をフラッシュさせる
    """

    import signal
    import sys
    from collections.abc import AsyncIterator

    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[listen] run_dir={run_dir}")
    typer.echo(f"[listen] docs_dir={docs_dir} (skip={skip_docs})")

    # ASR モジュールは [asr] extras 経由でしか入らない。遅延 import で
    # 「extras を入れていない人にもまともなエラーメッセージを出す」を実現。
    try:
        from das.asr import LiveAsrSession, build_engine
        from das.asr.mic import iter_mic_chunks
    except ImportError as exc:
        typer.echo(f"[listen] ASR 依存が未インストールです: {exc}")
        typer.echo("`uv sync --extra asr` を実行してください。")
        raise typer.Exit(1) from exc

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=run_dir / "graph.sqlite")
    orch = Orchestrator.assemble(
        llm=llm,
        store=store,
        threshold=threshold,
        top_k=top_k,
        top_k_per_source=top_k_per_source,
    )

    if not skip_docs and docs_dir.exists():
        typer.echo("[listen] ingesting documents...")
        await orch.ingest_documents(docs_dir)

    typer.echo(
        "[listen] loading ASR engine "
        "(初回はモデル DL で数十秒〜数分かかることがあります)..."
    )
    engine = build_engine(
        model=model,
        backend=backend,
        language=language,
        diarization=False,
    )

    stop_event = asyncio.Event()

    def _on_partial(text: str) -> None:
        # 同じ行を上書きで表示 (確定行は \n 付きで別行に出る)
        sys.stdout.write(f"\r... {text[:120]}")
        sys.stdout.flush()

    session = LiveAsrSession(engine=engine, on_partial=_on_partial)
    await session.start()

    # Ctrl-C で停止フラグを立てる。再度押されたら強制中断 (default 挙動に戻す)。
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        if not stop_event.is_set():
            typer.echo("\n[listen] 停止要求 (Ctrl-C)。残りをフラッシュ中...")
            stop_event.set()

    with contextlib.suppress(NotImplementedError):  # pragma: no cover - Windows
        loop.add_signal_handler(signal.SIGINT, _on_sigint)

    async def _pump_mic() -> None:
        async for chunk in iter_mic_chunks(stop_event=stop_event):
            await session.push_audio(chunk)
        # 入力が止まったら EOS を投げて ASR ジェネレータを終わりに向かわせる
        await session.stop()

    async def _utt_stream() -> AsyncIterator[Utterance]:
        async for utt in session.iter_utterances():
            sys.stdout.write(f"\r[t{utt.turn_id}] {utt.speaker}: {utt.text}\n")
            sys.stdout.flush()
            yield utt

    typer.echo("[listen] 録音開始。Ctrl-C で停止。")
    mic_task = asyncio.create_task(_pump_mic())
    try:
        await orch.run_live(_utt_stream())
    finally:
        # 異常終了でもマイクを確実に止める
        stop_event.set()
        try:
            await asyncio.wait_for(mic_task, timeout=10.0)
        except TimeoutError:
            mic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mic_task
        await session.cleanup()

    from das.viz import dump_snapshot

    snapshot_path = dump_snapshot(store, run_dir / "snapshot.json")
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))

    html_path: Path | None = None
    try:
        from das.viz import render_html

        html_path = render_html(store, run_dir / "graph.html")
    except ImportError as exc:
        typer.echo(
            f"[listen] HTML 生成をスキップ (viz extras 未インストール: {exc}). "
            f"`das visualize {snapshot_path}` で後から生成できます。"
        )

    summary = (
        f"\n[listen] done. nodes={n_nodes} edges={n_edges}\n"
        f"  snapshot -> {snapshot_path}"
    )
    if html_path is not None:
        summary += f"\n  html     -> {html_path}"
    typer.echo(summary)
