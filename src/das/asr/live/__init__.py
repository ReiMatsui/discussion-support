"""リアルタイム議事録ツール — エントリーポイント.

STTバックエンド（Soniox）経由で音声をストリーミングし、
声紋話者分離・AI ファシリテーション・議事録保存を統合する。

使い方:
  uv run python -m das.asr.live                     # マイクから文字起こし
  uv run python -m das.asr.live --debate 'テーマ'    # AI と1対1議論
  uv run python -m das.asr.live --simulate 'テーマ'  # AI同士の議論シミュレーション

das連携フック:
  ON_UTTERANCE に callable(speaker:str, text:str) を設定 →
  確定発話ごとに呼ばれる（cli.py がオーケストレータへ流すのに使用）。

パッケージ構成:
  _bootstrap.py       セッション初期化・起動ロジック
  _recv_loop.py       WebSocket受信 + flush（声紋判定・エコー除去）
  _workers.py         音声入力・送信・トピック抽出・ターンテイキング
  _session_state.py   共有状態 + ファイル出力
  _voice_profiles.py  声紋プロファイル
  _ui.py              HTTPサーバー + ターミナル出力
  _constants.py       定数・プロンプト・HTML テンプレート
  stt/                STTバックエンド
    __init__.py       STTBackend Protocol
    _soniox.py        Soniox 実装
  agents/             AIエージェント群
    _realtime.py      OpenAI Realtime API ファシリテーター
    _partner.py       Realtime API 議論パートナー
    _simulator.py     Chat+TTS シミュレーション
"""
from __future__ import annotations

import click

from das.asr.live._constants import _AGENT_TRIGGER

# ---------------------------------------------------------------------------
# das連携フック（外部API）
# ---------------------------------------------------------------------------

ON_UTTERANCE = None   # callable(speaker:str, text:str) | None
_SYS_HOOK_REF: list = [None]  # [callable | None] — _bootstrap から書き込まれる


def post_system(text: str) -> None:
    """das連携: ライブ議事録のタイムラインにシステム行(💡介入など)を外部から追加する."""
    hook = _SYS_HOOK_REF[0]
    if hook is not None:
        hook(text)


# ---------------------------------------------------------------------------
# CLI (click)
# ---------------------------------------------------------------------------

@click.command()
@click.option("--lang", default="ja", help="音声認識の言語")
@click.option("--model", default="stt-rt-v5", help="STTモデル名")
@click.option("--wav", default=None, type=click.Path(exists=False),
              help="指定で実マイクの代わりにファイル擬似ライブ")
@click.option("--play", is_flag=True, help="--wav使用時、スピーカーからも再生する")
@click.option("--join", is_flag=True,
              help="--wav使用時、再生しつつ自分のマイクも混ぜて参加する（イヤホン推奨）")
@click.option("--device", default=None, help="マイクデバイス名")
@click.option("--out", default=None, help="保存先mdファイル（省略時 transcripts/日時.md）")
@click.option("--no-open", is_flag=True, help="ブラウザを自動で開かない")
@click.option("--setup/--no-setup", default=True,
              help="起動時にブラウザで開始前設定を行う")
@click.option("--no-vp", is_flag=True, help="声紋照合を無効化")
@click.option("--voices", default="voices.json", help="声紋プロファイルの保存先")
@click.option("--vp-model", default="redimnet",
              type=click.Choice(["redimnet"]),
              help="声紋モデル（既定redimnet）")
@click.option("--vp-match", type=float, default=None,
              help="即時判定のしきい値（省略時はモデル別の既定値）")
@click.option("--vp-no-auto", is_flag=True, help="未知の声の自動登録を無効化")
@click.option("--vp-debug", is_flag=True, help="声紋判定の内訳を表示")
@click.option("--stt", default="soniox",
              type=click.Choice(["soniox"]),
              help="リアルタイムSTTの供給源")
@click.option("--soniox-endpoint/--no-soniox-endpoint", default=True,
              help="Sonioxのエンドポイント検出を使う")
@click.option("--diarization", default="none",
              type=click.Choice(["none", "pyannote"]),
              help="外部話者分離の供給源")
@click.option("--diarization-max-speakers", type=int, default=None,
              help="外部話者分離に渡す最大話者数ヒント")
@click.option("--vp-cluster-naming", is_flag=True,
              help="ハイブリッド構成: diarizationの生クラスタ単位で声紋照合し"
                   "名前を確定する（--diarization pyannote専用。声紋照合が"
                   "有効な時のみ機能。docs/design/pyannote_live1_trial_2026-07-09.md §9）")
@click.option("--vp-mint-cluster-link", is_flag=True,
              help="二重帳簿の根治(opt-in): 声紋が新しい人物Nを鋳造する瞬間に、"
                   "席を持つクラスタの蓄積声紋と対称比較し、同一人物なら新しい席を"
                   "作らずそのクラスタへ統合する（--vp-cluster-naming 併用時のみ。"
                   "docs/design/handoff_2026-07-25_dual_ledger_rootcure.md 案B）")
@click.option("--port", type=int, default=8231, help="UIサーバーのポート番号（0で無効）")
@click.option("--agent/--no-agent", default=True,
              help="AIファシリテーターを有効化（OPENAI_API_KEY必要）")
@click.option("--agent-voice", default="shimmer",
              help="AIファシリテーターの声")
@click.option("--agent-trigger", type=int, default=_AGENT_TRIGGER,
              help=f"AIの応答を検討する発話間隔（既定{_AGENT_TRIGGER}）")
@click.option("--simulate", metavar="TOPIC", default=None,
              help="AI議論シミュレーション（Chat+TTSで自動生成）")
@click.option("--sim-scenario", default=None,
              type=click.Choice(["stalled", "biased", "derailed",
                                 "consensus_needed", "healthy", "imbalanced"]),
              help="シミュレーションの議論パターン")
@click.option("--debate", metavar="TOPIC", default=None,
              help="AI会話相手と議論（Realtime APIで音声対話）")
@click.option("--debate-voice", default="echo", help="会話相手の声")
@click.option("--topic", metavar="TOPIC", default=None,
              help="人間同士の議論の議題（AI有効時の脱線判定の基準。"
                   "未指定なら会議冒頭から自動推定）")
@click.option("--proactivity", default="standard",
              type=click.Choice(["controlled", "standard", "active"]),
              help="ファシリテーターの介入の積極性（既定standard。"
                   "controlled=明確な問題時のみ）")
@click.option("--docs", default=None, metavar="DIR",
              help="--af 有効時に AF ランタイムが事前取り込みする文書ディレクトリ"
                   "（未指定なら取り込まない）")
@click.option("--af", is_flag=True,
              help="AF ベース介入を有効化（H1）。既定 OFF。毎発話 extraction+linking "
                   "が走るため重い。無指定ならルールベース介入のみ（恒久モード）。")
def main(**kwargs):
    """リアルタイム議事録 + AIファシリテーション."""
    from das.asr.live._bootstrap import LiveArgs, run_session

    # click のハイフン付きオプション名をアンダースコアに正規化
    mapped = {k.replace("-", "_"): v for k, v in kwargs.items()}
    args = LiveArgs(**mapped)
    run_session(args)


if __name__ == "__main__":
    main()
