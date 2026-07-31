"""
Shared pytest configuration.

`pyautogui` and `pywinctl` require a live display/session just to be
imported, which would make the `agent` package's test suite depend on
a real (or virtual, e.g. Xvfb) display. Since this project's own
policy is to keep tests fully offline and hermetic (see the mock
grounding-provider fixtures for the same rationale), we install fake
`pyautogui`/`pywinctl` modules into `sys.modules` before any test
module -- and therefore before any `agent.*` module -- is imported.

Individual tests can still reach into the fakes (e.g. to assert what
"OS calls" were dispatched) via the `fake_pyautogui` fixture in
`tests/conftest.py`.
"""

import sys

from tests.agent_fakes import make_fake_pyautogui, make_fake_pywinctl

_fake_pyautogui_module, _fake_recorder, _fake_mouse = make_fake_pyautogui()
_fake_pywinctl_module = make_fake_pywinctl()

sys.modules.setdefault("pyautogui", _fake_pyautogui_module)
sys.modules.setdefault("pywinctl", _fake_pywinctl_module)
