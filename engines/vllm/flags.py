from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Flag:
    name: str
    default: Any
    active: Any


FLAGS = {
    "chunked_prefill": Flag(
        name="enable_chunked_prefill",
        default=True,
        active=False,
    ),
    "prefix_caching": Flag(
        name="enable_prefix_caching",
        default=True,
        active=False,
    ),
}