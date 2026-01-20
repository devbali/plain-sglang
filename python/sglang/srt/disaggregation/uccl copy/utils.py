# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""UCCL-specific utilities (no custom memory pool support)."""

from typing import Any, Optional, Tuple


def init_uccl_custom_mem_pool(
    _device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    return False, None, None


def check_uccl_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    return False, None
