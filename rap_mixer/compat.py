"""Runtime compatibility helpers.

Hugging Face ZeroGPU Spaces currently pin Python 3.10, which lacks
``enum.StrEnum`` (added in Python 3.11). This fallback behaves the same for
our usage: members are strings and compare equal to their values.
"""

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
