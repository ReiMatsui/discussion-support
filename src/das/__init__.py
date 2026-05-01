"""Discussion Argumentation Support (das).

マルチエージェントによる議論グラフ統合型 議論支援システム。
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("discussion-support")
except PackageNotFoundError:  # pragma: no cover - editable install
    __version__ = "0.0.0"

__all__ = ["__version__"]
