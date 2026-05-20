"""칠러 물리 모델 단위 테스트 — COP 공식, 습구온도, 냉각 모드 전환."""

import pytest

from core.config.constants import (
    WET_BULB_FREE_THRESHOLD_C,
    WET_BULB_HYBRID_THRESHOLD_C,
    CHILLER_MAX_CAPACITY_KW,
)
from core.config.enums import CoolingMode
from domain.thermodynamics.chiller import (
    calculate_cop,
    calculate_wet_bulb_c,
    calculate_chiller_power_kw,
)


class TestWetBulb:
    def test_design_point(self):
        # T_wb ≈ T - (1 - RH/100) × (T - 4), RH=100% → T_wb = T
        assert calculate_wet_bulb_c(25.0, 100.0) == pytest.approx(25.0)

    def test_dry_air_lowers_wet_bulb(self):
        # 건조할수록 습구온도 < 건구온도
        wb = calculate_wet_bulb_c(30.0, 20.0)
        assert wb < 30.0

    def test_wet_bulb_never_exceeds_dry_bulb(self):
        # 공식 T_wb = T - (1 - RH/100) × (T - 4)는 T > 4°C에서만 T_wb ≤ T 성립.
        # T < 4°C는 (T-4)가 음수 → 근사식 특성상 역전 발생 (공식 한계 범위)
        for temp in [5.0, 10.0, 25.0, 40.0]:
            for humidity in [10.0, 50.0, 80.0, 100.0]:
                assert calculate_wet_bulb_c(temp, humidity) <= temp + 1e-9

    def test_formula_at_known_values(self):
        # T=20, RH=50%: T_wb = 20 - (1 - 0.5) × (20 - 4) = 20 - 8 = 12
        assert calculate_wet_bulb_c(20.0, 50.0) == pytest.approx(12.0)

    def test_humidity_100_returns_dry_bulb(self):
        # 포화 공기: 습구 = 건구
        temp = 15.0
        assert calculate_wet_bulb_c(temp, 100.0) == pytest.approx(temp)


class TestCOP:
    def test_design_point_is_6(self):
        # 설계 기준점: 외기 15°C, 공급 20°C → COP = 6.0
        assert calculate_cop(15.0, 20.0) == pytest.approx(6.0)

    def test_high_outdoor_temp_reduces_cop(self):
        # 외기 온도 오를수록 COP 감소
        cop_hot = calculate_cop(35.0, 20.0)
        cop_cold = calculate_cop(15.0, 20.0)
        assert cop_hot < cop_cold

    def test_outdoor_temp_sensitivity(self):
        # 외기 1°C 상승 → COP 0.1 감소
        cop_base = calculate_cop(15.0, 20.0)
        cop_plus1 = calculate_cop(16.0, 20.0)
        assert cop_base - cop_plus1 == pytest.approx(0.1, abs=1e-9)

    def test_higher_supply_temp_increases_cop(self):
        # 공급 온도 높을수록 냉동 사이클 리프트 감소 → COP 증가
        cop_low_supply = calculate_cop(25.0, 18.0)
        cop_high_supply = calculate_cop(25.0, 24.0)
        assert cop_high_supply > cop_low_supply

    def test_supply_temp_sensitivity(self):
        # 공급 온도 1°C 상승 → COP 0.25 증가
        cop_base = calculate_cop(15.0, 20.0)
        cop_plus1 = calculate_cop(15.0, 21.0)
        assert cop_plus1 - cop_base == pytest.approx(0.25, abs=1e-9)

    def test_minimum_cop_floor(self):
        # 극한 외기 온도에서도 COP ≥ 2.0
        assert calculate_cop(100.0, 18.0) == pytest.approx(2.0)
        assert calculate_cop(50.0, 18.0) >= 2.0

    def test_cop_always_positive(self):
        for outdoor in [-10.0, 0.0, 15.0, 30.0, 45.0]:
            assert calculate_cop(outdoor, 20.0) > 0.0


class TestChillerPowerAndMode:
    def test_free_cooling_mode_zero_chiller_power(self):
        # 습구온도 < 10°C → Free Cooling → 칠러 전력 0
        # T=5, RH=50% → T_wb = 5 - (1-0.5)*(5-4) = 5 - 0.5 = 4.5°C < 10
        result = calculate_chiller_power_kw(
            cooling_load_kw=100.0,
            outdoor_temp_c=5.0,
            supply_temp_c=20.0,
            outdoor_humidity_pct=50.0,
        )
        assert result.cooling_mode == CoolingMode.FREE_COOLING
        assert result.chiller_power_kw == pytest.approx(0.0)

    def test_chiller_only_mode(self):
        # 습구온도 > 18°C → Chiller Only
        # T=30, RH=70% → T_wb = 30 - (1-0.7)*(30-4) = 30 - 7.8 = 22.2°C > 18
        result = calculate_chiller_power_kw(
            cooling_load_kw=100.0,
            outdoor_temp_c=30.0,
            supply_temp_c=20.0,
            outdoor_humidity_pct=70.0,
        )
        assert result.cooling_mode == CoolingMode.CHILLER
        assert result.chiller_power_kw > 0.0

    def test_hybrid_mode(self):
        # 습구온도 10~18°C → Hybrid
        # T=20, RH=50% → T_wb = 12°C (테스트에서 이미 검증)
        result = calculate_chiller_power_kw(
            cooling_load_kw=100.0,
            outdoor_temp_c=20.0,
            supply_temp_c=20.0,
            outdoor_humidity_pct=50.0,
        )
        assert result.cooling_mode == CoolingMode.HYBRID
        assert 0.0 < result.chiller_power_kw < (100.0 / result.cop)

    def test_chiller_power_equals_load_div_cop_in_chiller_mode(self):
        # Chiller 모드: P_chiller = Q / COP (PLR 보정 적용)
        # PLR 보정 제거 후 순수 검증하기 위해 부하를 최적 PLR에 맞춤
        # CHILLER_MAX_CAPACITY_KW = 250, 최적 PLR = 0.6 → 최적 부하 = 150 kW
        optimal_load = CHILLER_MAX_CAPACITY_KW * 0.6
        result = calculate_chiller_power_kw(
            cooling_load_kw=optimal_load,
            outdoor_temp_c=30.0,
            outdoor_humidity_pct=80.0,
            supply_temp_c=20.0,
        )
        assert result.cooling_mode == CoolingMode.CHILLER
        # PLR = 최적값이므로 plr_factor ≈ 1.0
        cop_base = calculate_cop(30.0, 20.0)
        expected_power = optimal_load / (cop_base * 1.0)
        assert result.chiller_power_kw == pytest.approx(expected_power, rel=0.05)

    def test_negative_cooling_load_raises(self):
        with pytest.raises(ValueError, match="0 이상"):
            calculate_chiller_power_kw(-10.0, 25.0)

    def test_cop_result_always_positive(self):
        for outdoor in [5.0, 20.0, 35.0]:
            result = calculate_chiller_power_kw(100.0, outdoor)
            assert result.cop > 0.0
