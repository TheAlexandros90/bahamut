from .split import Bahamut, BahamutSplit, BahamutSplitConfig, Bahamut_split, demo_bahamut_split
from . import interactive as _interactive

__all__ = [
    "Bahamut",
    "BahamutSplit",
    "BahamutSplitConfig",
    "Bahamut_split",
    "demo_bahamut_split",
]

_interactive.attach_interactive_api(BahamutSplit)
