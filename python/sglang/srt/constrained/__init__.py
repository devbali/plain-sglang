try:
    from outlines.caching import cache as disk_cache
    from outlines.caching import disable_cache
    from outlines.fsm.guide import RegexGuide
    from outlines.fsm.regex import FSMInfo, make_byte_level_fsm, make_deterministic_fsm
    from outlines.models.transformers import TransformerTokenizer
except ImportError:
    def disable_cache(): pass
