"""
Unit tests for `agent.tools.action.actions.ActionManager`.

All OS interaction goes through the fake `pyautogui` module installed
by the root `conftest.py`, so these tests run fully offline with no
display dependency.
"""

import pytest

from agent.tools.action.action_models import MousePosition
from agent.tools.action.actions import ActionManager
from agent.tools.coordinates import CoordinateMapper, IdentityCoordinateMapper
from grounding.models import BoundingBox, GroundingDetection
from tests.agent_fakes import CallRecorder, MouseState


def make_manager(
    *, physical: tuple[int, int] = (1920, 1080), logical: tuple[int, int] | None = None
) -> ActionManager:
    logical = physical if logical is None else logical

    def physical_size_fn() -> tuple[int, int]:
        return physical

    def logical_size_fn() -> tuple[int, int]:
        return logical

    mapper = CoordinateMapper(
        physical_size_fn=physical_size_fn,
        logical_size_fn=logical_size_fn,
    )
    return ActionManager(post_action_delay=0.0, coordinate_mapper=mapper)


def make_detection(
    x1: int = 100, y1: int = 100, x2: int = 200, y2: int = 200
) -> GroundingDetection:
    return GroundingDetection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
        label="button",
    )


class TestResolvePosition:
    def test_resolves_tuple_target_at_1x(self) -> None:
        manager = make_manager()

        position = manager._resolve_position((150, 200))

        assert position == MousePosition(x=150, y=200)

    def test_resolves_tuple_target_with_dpi_scaling(self) -> None:
        # 2x Retina: screenshot pixel (300, 400) -> logical (150, 200)
        manager = make_manager(physical=(2880, 1800), logical=(1440, 900))

        position = manager._resolve_position((300, 400))

        assert position == MousePosition(x=150, y=200)

    def test_resolves_bounding_box_center(self) -> None:
        manager = make_manager()

        position = manager._resolve_position(
            BoundingBox(x1=100, y1=100, x2=200, y2=300)
        )

        assert position == MousePosition(x=150, y=200)

    def test_resolves_grounding_detection_center_with_dpi_scaling(
        self,
    ) -> None:
        manager = make_manager(physical=(2880, 1800), logical=(1440, 900))
        detection = make_detection(x1=0, y1=0, x2=400, y2=200)  # center 200,100

        position = manager._resolve_position(detection)

        assert position == MousePosition(x=100, y=50)

    def test_invalid_tuple_length_raises(self) -> None:
        manager = make_manager()

        with pytest.raises(ValueError):
            manager._resolve_position((1, 2, 3))  # type: ignore[arg-type]

    def test_unsupported_target_type_raises(self) -> None:
        manager = make_manager()

        with pytest.raises(TypeError):
            manager._resolve_position(object())  # type: ignore[arg-type]


class TestMouseActions:
    def test_click_dispatches_to_pyautogui(
        self,
        fake_pyautogui_recorder: CallRecorder,
        fake_mouse: MouseState,
    ) -> None:
        manager = make_manager()

        result = manager.click((100, 200))

        assert result.success
        assert result.action == "click"
        assert result.position == MousePosition(x=100, y=200)
        click_calls = fake_pyautogui_recorder.calls_named("click")
        assert len(click_calls) == 1
        _, kwargs = click_calls[0]
        assert kwargs["x"] == 100
        assert kwargs["y"] == 200

    def test_click_records_latency(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.click((10, 10))

        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    def test_click_applies_dpi_scaling_before_dispatch(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager(physical=(2880, 1800), logical=(1440, 900))

        manager.click((2400, 1200))

        _, kwargs = fake_pyautogui_recorder.calls_named("click")[0]
        assert kwargs["x"] == 1200
        assert kwargs["y"] == 600

    def test_double_click_sends_two_clicks(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        manager.double_click((10, 10))

        _, kwargs = fake_pyautogui_recorder.calls_named("click")[0]
        assert kwargs["clicks"] == 2

    def test_right_click_uses_right_button(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        manager.right_click((10, 10))

        _, kwargs = fake_pyautogui_recorder.calls_named("click")[0]
        assert kwargs["button"] == "right"

    def test_move_mouse_moves_without_clicking(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        result = manager.move_mouse((50, 60))

        assert result.action == "move_mouse"
        assert fake_pyautogui_recorder.calls_named("click") == []
        assert len(fake_pyautogui_recorder.calls_named("moveTo")) == 1

    def test_drag_from_explicit_start_to_end(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        result = manager.drag(start=(0, 0), end=(100, 100))

        assert result.action == "drag"
        assert len(fake_pyautogui_recorder.calls_named("dragTo")) == 1

    def test_drag_detection_to_detection(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()
        source = make_detection(0, 0, 20, 20)
        target = make_detection(80, 80, 100, 100)

        result = manager.drag_detection_to_detection(source=source, target=target)

        assert result.action == "drag"

    def test_scroll_without_target(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.scroll(-5)

        assert result.position is None
        assert fake_pyautogui_recorder.calls_named("scroll") == [((), {"clicks": -5})]

    def test_scroll_with_target_moves_first(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        result = manager.scroll(3, target=(10, 10))

        assert result.position == MousePosition(x=10, y=10)

    def test_hscroll(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.hscroll(4)

        assert result.success
        assert fake_pyautogui_recorder.calls_named("hscroll") == [((), {"clicks": 4})]


class TestKeyboardActions:
    def test_type_text(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.type_text("hello world")

        assert result.action == "type_text"
        assert result.message == "hello world"
        assert fake_pyautogui_recorder.calls_named("write") == [
            ((), {"text": "hello world", "interval": 0.0})
        ]

    def test_press_key(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.press_key("enter", presses=2)

        assert result.action == "press_key"
        _, kwargs = fake_pyautogui_recorder.calls_named("press")[0]
        assert kwargs["presses"] == 2

    def test_hotkey(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.hotkey("ctrl", "c")

        assert result.message == "ctrl + c"
        _, kwargs = fake_pyautogui_recorder.calls_named("hotkey")[0]
        assert kwargs["keys"] == ("ctrl", "c")

    def test_hold_key_presses_and_releases(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        with manager.hold_key("shift"):
            pass

        assert fake_pyautogui_recorder.calls_named("keyDown") == [
            ((), {"key": "shift"})
        ]
        assert fake_pyautogui_recorder.calls_named("keyUp") == [((), {"key": "shift"})]

    def test_hold_key_releases_even_on_exception(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        with pytest.raises(RuntimeError), manager.hold_key("shift"):
            raise RuntimeError("boom")

        assert fake_pyautogui_recorder.calls_named("keyUp") == [((), {"key": "shift"})]

    def test_with_modifier_wraps_action_in_hold_key(
        self, fake_pyautogui_recorder: CallRecorder
    ) -> None:
        manager = make_manager()

        result = manager.with_modifier("ctrl", lambda: manager.click((5, 5)))

        assert result.action == "click"
        assert fake_pyautogui_recorder.calls_named("keyDown") == [((), {"key": "ctrl"})]


class TestOtherActions:
    def test_delay_sleeps_for_given_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = make_manager()
        slept: list[float] = []

        def fake_sleep(s: float) -> None:
            slept.append(s)

        monkeypatch.setattr(
            "agent.tools.action.actions.time.sleep",
            fake_sleep,
        )

        result = manager.delay(0.25)

        assert result.action == "wait"
        assert 0.25 in slept

    def test_delay_rejects_negative_seconds(self) -> None:
        manager = make_manager()

        with pytest.raises(ValueError):
            manager.delay(-1)

    def test_click_and_type(self, fake_pyautogui_recorder: CallRecorder) -> None:
        manager = make_manager()

        result = manager.click_and_type((10, 10), "hi")

        assert result.action == "type_text"
        assert "hi" in (result.message or "")
        assert fake_pyautogui_recorder.calls_named("click")
        assert fake_pyautogui_recorder.calls_named("write")


class TestUtility:
    def test_mouse_position(self, fake_mouse: MouseState) -> None:
        fake_mouse.x, fake_mouse.y = 42, 24

        assert ActionManager.mouse_position() == MousePosition(x=42, y=24)

    def test_screen_size(self) -> None:
        assert ActionManager.screen_size() == (1440, 900)

    def test_close_is_a_noop(self) -> None:
        manager = make_manager()
        manager.close()  # should not raise

    def test_default_coordinate_mapper_uses_pyautogui(self) -> None:
        manager = ActionManager(post_action_delay=0.0)

        assert manager._coordinate_mapper.scale.is_identity

    def test_identity_mapper_can_be_supplied_explicitly(self) -> None:
        manager = ActionManager(
            post_action_delay=0.0,
            coordinate_mapper=IdentityCoordinateMapper(),
        )

        assert manager._resolve_position((999, 888)) == MousePosition(x=999, y=888)
