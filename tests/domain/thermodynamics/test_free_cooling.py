"""자연공조(Free Cooling) 모델 단위 테스트 — 효율, 팬 전력, 냉각 모드 분기."""

import pytest

from core.config.constants import (
    WET_BULB_FREE_THRESHOLD_C,
    WET_BULB_HYBRID_THRESHOLD_C,
    FAN_POWER_RATIO_FREE,
    FAN_POWER_RATIO_CHILLER,
)
from domain.thermodynamics.free_cooling import (
    calculate_free_cooling_efficiency,
    calculate_free_cooling,
    FreeCoolingResult,
)


class TestFreeCoolingEfficiency:
    def test_full_efficiency_below_free_threshold(self):
        # wet_bulb < 10°C → 완전 자연공조, efficiency ≈ 1.0
        # T=5, RH=50% → T_wb = 5 - 0.5*(5-4) = 4.5°C < 10
        eff = calculate_free_cooling_efficiency(5.0, outdoor_humidity_pct=50.0)
        assert eff == pytest.approx(1.0, abs=0.05)

    def test_zero_efficiency_above_hybrid_threshold(self):
        # wet_bulb > 18°C → 기계식 전환, efficiency = 0.0
        # T=30, RH=80% → T_wb = 30 - 0.2*(30-4) = 24.8°C > 18
        eff = calculate_free_cooling_efficiency(30.0, outdoor_humidity_pct=80.0)
        assert eff == pytest.approx(0.0)

    def test_partial_efficiency_in_hybrid_zone(self):
        # wet_bulb 10~18°C → 0 < efficiency < 1
        # T=20, RH=50% → T_wb = 12°C (혼합 구간)
        eff = calculate_free_cooling_efficiency(20.0, outdoor_humidity_pct=50.0)
        assert 0.0 < eff < 1.0

    def test_efficiency_decreases_with_higher_wet_bulb(self):
        # wet_bulb 높을수록 효율 감소
        eff_low = calculate_free_cooling_efficiency(10.0, outdoor_humidity_pct=50.0)
        eff_high = calculate_free_cooling_efficiency(25.0, outdoor_humidity_pct=50.0)
        assert eff_low > eff_high

    def test_efficiency_bounded_zero_to_one(self):
        for temp in [-10.0, 0.0, 10.0, 20.0, 35.0, 45.0]:
            eff = calculate_free_cooling_efficiency(temp, 50.0)
            assert 0.0 <= eff <= 1.0

    def test_hybrid_boundary_linear_interpolation(self):
        # wet_bulb = 14°C (중간점): efficiency ≈ 0.5 (선형 보간)
        # T=20, RH=50% → T_wb = 12°C이므로 T=22, RH=50% → T_wb 확인 필요
        # 직접: T=18, RH=50% → T_wb = 18 - 0.5*(18-4) = 18 - 7 = 11°C
        # T=24, RH=50% → T_wb = 24 - 0.5*(24-4) = 24 - 10 = 14°C ← 중간점
        eff = calculate_free_cooling_efficiency(24.0, outdoor_humidity_pct=50.0)
        assert 0.3 < eff < 0.7


class TestFreeCoolingResult:
    def test_free_cooling_available_in_cold_weather(self):
        # 완전 자연공조: is_available = True
        result = calculate_free_cooling(100.0, outdoor_temp_c=5.0, outdoor_humidity_pct=50.0)
        assert result.is_available is True

    def test_free_cooling_unavailable_in_hot_weather(self):
        # 기계식 냉방: is_available = False
        result = calculate_free_cooling(100.0, outdoor_temp_c=35.0, outdoor_humidity_pct=80.0)
        assert result.is_available is False

    def test_effective_cooling_equals_load_times_efficiency(self):
        load = 150.0
        result = calculate_free_cooling(load, outdoor_temp_c=5.0, outdoor_humidity_pct=50.0)
        assert result.effective_cooling_kw == pytest.approx(load * result.efficiency, rel=0.01)

    def test_fan_power_positive(self):
        result = calculate_free_cooling(100.0, outdoor_temp_c=10.0, outdoor_humidity_pct=60.0)
        assert result.fan_power_kw > 0.0

    def test_full_free_cooling_fan_power(self):
        # 완전 자연공조(efficiency=1): fan = load × FAN_POWER_RATIO_FREE
        load = 100.0
        result = calculate_free_cooling(load, outdoor_temp_c=0.0, outdoor_humidity_pct=10.0)
        expected_fan = load * (FAN_POWER_RATIO_FREE * 1.0 + FAN_POWER_RATIO_CHILLER * 0.0)
        assert result.fan_power_kw == pytest.approx(expected_fan, rel=0.05)

    def test_full_chiller_fan_power(self):
        # 기계식 전용(efficiency=0): fan = load × FAN_POWER_RATIO_CHILLER
        load = 100.0
        result = calculate_free_cooling(load, outdoor_temp_c=35.0, outdoor_humidity_pct=80.0)
        expected_fan = load * (FAN_POWER_RATIO_FREE * 0.0 + FAN_POWER_RATIO_CHILLER * 1.0)
        assert result.fan_power_kw == pytest.approx(expected_fan, rel=0.05)

    def test_negative_cooling_load_raises(self):
        with pytest.raises(ValueError, match="0 이상"):
            calculate_free_cooling(-10.0, outdoor_temp_c=10.0)

    def test_result_is_dataclass(self):
        result = calculate_free_cooling(100.0, outdoor_temp_c=15.0)
        assert isinstance(result, FreeCoolingResult)

    def test_mode_description_contains_temp(self):
        result = calculate_free_cooling(100.0, outdoor_temp_c=15.0)
        assert "15.0" in result.mode_description
