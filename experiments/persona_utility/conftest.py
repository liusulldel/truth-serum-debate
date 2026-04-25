# Force importlib mode so each subdir's `mock_persona_debater.py` /
# `experiment.py` resolves to its own module rather than colliding on
# unqualified module names. (pytest default 'prepend' mode caches the
# first sibling it imports under that bare name.)
collect_ignore_glob: list[str] = []
