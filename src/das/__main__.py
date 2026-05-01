"""``python -m das`` のエントリポイント。"""

from das.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
