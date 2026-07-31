"""
Unit tests for `agent.tools.coordinates.CoordinateMapper`.

These tests cover the High-DPI / Retina coordinate scaling bug fix:
screenshot-space pixel coordinates must be converted to OS logical
coordinates before being handed to `pyautogui`.
"""

import pytest

from agent.tools.coordinates import CoordinateMapper, IdentityCoordinateMapper


def make_mapper(
    *, physical: tuple[int, int], logical: tuple[int, int]
) -> CoordinateMapper:
    return CoordinateMapper(
        physical_size_fn=lambda: physical,
        logical_size_fn=lambda: logical,
    )


class TestScaleComputation:
    def test_identity_scale_when_sizes_match(self) -> None:
        mapper = make_mapper(physical=(1920, 1080), logical=(1920, 1080))

        scale = mapper.scale

        assert scale.x == pytest.approx(1.0)
        assert scale.y == pytest.approx(1.0)
        assert scale.is_identity

    def test_2x_retina_scale(self) -> None:
        mapper = make_mapper(physical=(2880, 1800), logical=(1440, 900))

        scale = mapper.scale

        assert scale.x == pytest.approx(2.0)
        assert scale.y == pytest.approx(2.0)
        assert not scale.is_identity

    def test_asymmetric_scale(self) -> None:
        mapper = make_mapper(physical=(3000, 1500), logical=(1500, 1000))

        scale = mapper.scale

        assert scale.x == pytest.approx(2.0)
        assert scale.y == pytest.approx(1.5)

    def test_scale_is_cached(self) -> None:
        calls = {"count": 0}

        def physical_size_fn() -> tuple[int, int]:
            calls["count"] += 1
            return (1920, 1080)

        mapper = CoordinateMapper(
            physical_size_fn=physical_size_fn,
            logical_size_fn=lambda: (1920, 1080),
        )

        mapper.scale
        mapper.scale
        mapper.scale

        assert calls["count"] == 1

    def test_refresh_recomputes_scale(self) -> None:
        sizes = iter([(1920, 1080), (3840, 2160)])

        mapper = CoordinateMapper(
            physical_size_fn=lambda: next(sizes),
            logical_size_fn=lambda: (1920, 1080),
        )

        first = mapper.scale
        second = mapper.refresh()

        assert first.x == pytest.approx(1.0)
        assert second.x == pytest.approx(2.0)

    def test_non_positive_logical_size_raises(self) -> None:
        mapper = make_mapper(physical=(1920, 1080), logical=(0, 1080))

        with pytest.raises(ValueError):
            mapper.scale


class TestCoordinateConversion:
    def test_to_logical_is_noop_at_1x(self) -> None:
        mapper = make_mapper(physical=(1920, 1080), logical=(1920, 1080))

        assert mapper.to_logical(500, 300) == (500, 300)

    def test_to_logical_scales_down_on_retina(self) -> None:
        # A grounding model reports a bbox center of (2400, 1200) in
        # screenshot pixel space on a 2x Retina display. The real OS
        # logical coordinate that must be clicked is (1200, 600).
        mapper = make_mapper(physical=(2880, 1800), logical=(1440, 900))

        assert mapper.to_logical(2400, 1200) == (1200, 600)

    def test_to_logical_rounds_to_nearest_int(self) -> None:
        mapper = make_mapper(physical=(3000, 1500), logical=(2000, 1000))

        # scale = 1.5x; 100 / 1.5 = 66.66... -> rounds to 67
        assert mapper.to_logical(100, 100) == (67, 67)

    def test_to_physical_is_inverse_of_to_logical(self) -> None:
        mapper = make_mapper(physical=(2880, 1800), logical=(1440, 900))

        logical = mapper.to_logical(2400, 1200)
        physical = mapper.to_physical(*logical)

        assert physical == (2400, 1200)


class TestIdentityCoordinateMapper:
    def test_never_scales(self) -> None:
        mapper = IdentityCoordinateMapper()

        assert mapper.to_logical(1234, 5678) == (1234, 5678)
        assert mapper.scale.is_identity

    def test_refresh_stays_identity(self) -> None:
        mapper = IdentityCoordinateMapper()

        assert mapper.refresh().is_identity
