"""냉각 부하 물리 모델 단위 테스트 — Q = ṁcₚΔT, 에너지 보존 법칙."""

import pytest

from domain.thermodynamics.cooling_load import (
    calculate_cooling_load_from_airflow_kw,
    calculate_cooling_load_from_it_power_kw,
    calculate_m_air_for_servers,
    AIR_SPECIFIC_HEAT_KJ_PER_KG_K,
)
from core.config.constants import M_AIR_DESIGN_KG_S, NUM_SERVERS_DESIGN


class TestAirflowCoolingLoad:
    def test_q_equals_m_cp_delta_t(self):
        # Q = ṁ × c_p × ΔT
        m = 33.0    # kg/s
        cp = AIR_SPECIFIC_HEAT_KJ_PER_KG_K
        delta_t = 7.0  # °C (공급 20 → 환기 27)
        expected = m * cp * delta_t
        result = calculate_cooling_load_from_airflow_kw(
            m_dot_kg_per_s=m,
            supply_temp_c=20.0,
            return_temp_c=27.0,
        )
        assert result == pytest.approx(expected)

    def test_design_point(self):
        # 설계값: ṁ=33 kg/s, 공급 20°C, 환기 27°C → Q ≈ 232 kW
        result = calculate_cooling_load_from_airflow_kw(
            m_dot_kg_per_s=M_AIR_DESIGN_KG_S,
            supply_temp_c=20.0,
            return_temp_c=27.0,
        )
        assert result == pytest.approx(M_AIR_DESIGN_KG_S * 1.005 * 7.0, abs=0.5)

    def test_larger_delta_t_gives_more_load(self):
        common = dict(m_dot_kg_per_s=33.0, supply_temp_c=20.0)
        low = calculate_cooling_load_from_airflow_kw(**common, return_temp_c=24.0)
        high = calculate_cooling_load_from_airflow_kw(**common, return_temp_c=30.0)
        assert high > low

    def test_return_below_supply_raises(self):
        # 환기 온도 < 공급 온도: 물리적으로 불가
        with pytest.raises(ValueError, match="낮습니다"):
            calculate_cooling_load_from_airflow_kw(33.0, supply_temp_c=22.0, return_temp_c=20.0)

    def test_equal_temps_gives_zero_load(self):
        result = calculate_cooling_load_from_airflow_kw(33.0, 20.0, 20.0)
        assert result == pytest.approx(0.0)

    def test_custom_cp(self):
        result = calculate_cooling_load_from_airflow_kw(10.0, 20.0, 25.0, c_p=2.0)
        assert result == pytest.approx(10.0 * 2.0 * 5.0)


class TestITPowerCoolingLoad:
    def test_overhead_factor_one_returns_it_power(self):
        # 에너지 보존: 냉각 부하 = IT 전력 (overhead_factor=1.0)
        assert calculate_cooling_load_from_it_power_kw(150.0) == pytest.approx(150.0)

    def test_overhead_factor_applied(self):
        result = calculate_cooling_load_from_it_power_kw(100.0, overhead_factor=1.05)
        assert result == pytest.approx(105.0)

    def test_negative_it_power_raises(self):
        with pytest.raises(ValueError, match="0 이상"):
            calculate_cooling_load_from_it_power_kw(-10.0)

    def test_zero_it_power_returns_zero(self):
        assert calculate_cooling_load_from_it_power_kw(0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("power_kw", [50.0, 100.0, 200.0, 300.0])
    def test_result_nonnegative(self, power_kw):
        assert calculate_cooling_load_from_it_power_kw(power_kw) >= 0.0


class TestMAirScaling:
    def test_design_server_count_returns_design_flow(self):
        # 설계 서버 수 → 설계 풍량
        result = calculate_m_air_for_servers(NUM_SERVERS_DESIGN)
        assert result == pytest.approx(M_AIR_DESIGN_KG_S)

    def test_linear_scaling(self):
        # 서버 수 2배 → 풍량 2배
        half = calculate_m_air_for_servers(NUM_SERVERS_DESIGN // 2)
        full = calculate_m_air_for_servers(NUM_SERVERS_DESIGN)
        assert full == pytest.approx(half * 2.0, rel=0.01)

    def test_zero_servers_gives_zero_flow(self):
        assert calculate_m_air_for_servers(0) == pytest.approx(0.0)
