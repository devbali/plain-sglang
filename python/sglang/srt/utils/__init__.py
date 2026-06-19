# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *

import importlib.util as _iu, os as _os
_legacy_path = _os.path.join(_os.path.dirname(__file__), "..", "utils.py")
if _os.path.exists(_legacy_path):
    _spec = _iu.spec_from_file_location("sglang.srt._utils_legacy", _legacy_path)
    _legacy = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_legacy)
    configure_logger = _legacy.configure_logger
    kill_parent_process = _legacy.kill_parent_process
