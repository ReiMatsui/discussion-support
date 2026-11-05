"""旧接続確認の入口。使用量制限付き検査へ統合（既定はAPIに送信しない）。"""

from .bounded_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
