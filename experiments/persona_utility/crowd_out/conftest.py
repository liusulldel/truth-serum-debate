import os, sys
# Drop sibling subdirs' cached bare-name modules so each subdir's
# `from experiment import ...` and `from mock_persona_debater import ...`
# resolves to ITS OWN local file when tests are collected together.
for _name in ("experiment", "mock_persona_debater"):
    sys.modules.pop(_name, None)
sys.path.insert(0, os.path.dirname(__file__))
