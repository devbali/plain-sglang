"""UCCL-specific utilities (no custom memory pool support)."""

from typing import Any, Optional, Tuple


def init_uccl_custom_mem_pool(
    _device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    return False, None, None


def check_uccl_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    return False, None
