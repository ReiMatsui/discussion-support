"""常駐処理の責務: lifecycle。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._ui import _print_line


def _run_stdin_commands(state: SessionState):
    """標準入力からの話者リネーム・統合コマンドを処理."""
    while not state.stop.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
        m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
        if mfix:
            src, dst = state.key_of(mfix.group(1)), state.key_of(mfix.group(2))
            if state.tracker is not None:
                state.tracker.remap(src, dst)
            state.rekey(src, dst)
            state.add_sys(None, f"{state.disp_name(src)} を {state.disp_name(dst)} に統合（手動fix）")
            state.save()
            _print_line(f"# {state.disp_name(src)} を {state.disp_name(dst)} に統合しました（過去の発言も修正済み）")
        elif m:
            label, name = m.group(1), m.group(2)
            if state.tracker is not None:
                old = state.tracker.enroll(label, name)
                if old is None:
                    _print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                    continue
                state.rekey(old, name)
                state.add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                state.save()
                _print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
            else:
                state.set_display_name("#" + label, name)
                state.save()
                _print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
        elif line.strip():
            _print_line("# 名前登録はブラウザUIを推奨。ターミナル操作: 「1=名前」/「fix 2=1」/ Ctrl+Cで終了")


def _cleanup(state: SessionState, tracker, wav_path: str, out_path: str, html_path: str):
    """セッション終了時のリソース解放・ファイル保存."""
    from das.asr.live import _SYS_HOOK_REF
    _SYS_HOOK_REF[0] = None
    state.stop.set()
    if state.partner is not None:
        state.partner.close()
    if state.simulator is not None:
        state.simulator.shutdown()
    if state.agent is not None:
        state.agent.close()
    state.save(live=False)
    if tracker is not None:
        _print_line(f"# レイテンシ統計: {tracker.stats()}")
    saved = state.out_path if state._serve else f"{state.out_path} / {state.html_path}"
    _print_line(f"# 議事録を保存しました: {saved}")
    # WAVファイルのヘッダを確定して正規のWAVにする（state.wav_pathを使用）
    saved_wav = state.finalize_wav()
    if saved_wav:
        _print_line(f"# 録音を保存しました: {saved_wav}")

