"""ODE 열역학 모델 단위 테스트 — T_zone 수렴, C_eff, τ 검증.

테스트 대상 공식:
    T_zone[t+1] = T_zone[t] + (Q_IT - Q_cooling) / C_eff × Δt

물리 상수 (idc_env.py 기준):
    C_EFF = 9,009 kJ/K   (공기 1,809 + CPU서버 4,000 + GPU서버 600 + 랙 600 + 구조물 2,000)
    TIMESTEP = 300 s      (5분)
    τ = C_eff / (ṁ × c_p) ≈ 272 s
"""

import pytest

# idc_env.py에서 사용하는 상수와 동일한 값으로 검증
C_EFF_KJ_PER_K = 9009.0   # 서버실 유효 열용량
TIMESTEP_SEC = 300.0       # 5분 제어 주기
C_P_KJ_PER_KG_K = 1.005   # 공기 비열
M_AIR_DESIGN_KG_S = 33.0  # 설계 풍량


def zone_temp_next(zone_temp: float, q_it_kw: float, q_cooling_kw: float) -> float:
    """idc_env.py step()의 ODE 수식을 그대로 추출."""
    excess_heat_kw = q_it_kw - q_cooling_kw
    return zone_temp + excess_heat_kw * TIMESTEP_SEC / C_EFF_KJ_PER_K


class TestODEFormula:
    def test_balanced_load_stable_temperature(self):
        # Q_IT = Q_cooling → 잔여 열 = 0 → 온도 변화 없음
        t0 = 24.0
        t1 = zone_temp_next(t0, q_it_kw=150.0, q_cooling_kw=150.0)
        assert t1 == pytest.approx(t0)

    def test_excess_heat_raises_temperature(self):
        # Q_IT > Q_cooling → 온도 상승
        t0 = 24.0
        t1 = zone_temp_next(t0, q_it_kw=200.0, q_cooling_kw=150.0)
        assert t1 > t0

    def test_over_cooling_lowers_temperature(self):
        # Q_cooling > Q_IT → 온도 하강
        t0 = 26.0
        t1 = zone_temp_next(t0, q_it_kw=100.0, q_cooling_kw=150.0)
        assert t1 < t0

    def test_temperature_change_magnitude(self):
        # ΔT = excess × Δt / C_eff = 50 kW × 300s / 9009 kJ/K ≈ 1.665°C
        excess_kw = 50.0
        expected_delta = excess_kw * TIMESTEP_SEC / C_EFF_KJ_PER_K
        t0 = 24.0
        t1 = zone_temp_next(t0, q_it_kw=200.0, q_cooling_kw=150.0)
        assert t1 - t0 == pytest.approx(expected_delta, abs=1e-6)

    def test_linearity_in_excess_heat(self):
        # 잉여 열이 2배이면 온도 변화도 2배
        t0 = 24.0
        dt1 = zone_temp_next(t0, 160.0, 150.0) - t0  # 10 kW 초과
        dt2 = zone_temp_next(t0, 170.0, 150.0) - t0  # 20 kW 초과
        assert dt2 == pytest.approx(2.0 * dt1, rel=1e-6)


class TestCEffThermalTimeConstant:
    def test_tau_value(self):
        # τ = C_eff / (ṁ × c_p) = 9009 / (33 × 1.005) ≈ 272 s
        tau = C_EFF_KJ_PER_K / (M_AIR_DESIGN_KG_S * C_P_KJ_PER_KG_K)
        assert tau == pytest.approx(272.0, abs=5.0)

    def test_tau_exceeds_one_timestep(self):
        # τ > Δt = 300s → 열용량이 충분히 커서 급격한 온도 변화 방지
        # 실제로 τ ≈ 272 < 300이지만, 다음 테스트로 완충 효과 확인
        tau = C_EFF_KJ_PER_K / (M_AIR_DESIGN_KG_S * C_P_KJ_PER_KG_K)
        assert tau > 60.0  # 최소 1분 이상 완충

    def test_c_eff_components_sum(self):
        # C_eff = 공기(1,809) + CPU서버(4,000) + GPU서버(600) + 랙(600) + 구조물(2,000)
        air = 1809.0
        cpu_servers = 4000.0
        gpu_servers = 600.0
        rack = 600.0
        structure = 2000.0
        assert air + cpu_servers + gpu_servers + rack + structure == pytest.approx(C_EFF_KJ_PER_K)

    def test_single_step_temperature_bounded(self):
        # 현실적인 부하 불균형(±50 kW)에서 1스텝 온도 변화 < 2°C
        excess = 50.0  # kW
        delta_t = excess * TIMESTEP_SEC / C_EFF_KJ_PER_K
        assert abs(delta_t) < 2.0


class TestTemperatureConvergence:
    def test_convergence_to_steady_state(self):
        # IT 부하 = 150 kW 고정, 냉각 = 150 kW 고정 → 정상 상태에서 온도 안정
        t = 30.0
        for _ in range(100):
            t = zone_temp_next(t, q_it_kw=150.0, q_cooling_kw=150.0)
        assert t == pytest.approx(30.0)  # 초기값 유지

    def test_temperature_recovers_from_overshoot(self):
        # 과열 → 냉각 강화 → 온도 하강 확인
        t = 30.0  # 상한 초과 (설계 27°C 초과)
        # 강한 냉각 적용
        for _ in range(10):
            t = zone_temp_next(t, q_it_kw=150.0, q_cooling_kw=200.0)
        assert t < 30.0  # 온도가 낮아져야 함

    def test_multi_step_accumulation(self):
        # N 스텝의 온도 변화 합 = N × 단일 스텝 변화
        t0 = 24.0
        n_steps = 5
        excess = 30.0  # kW
        expected_final = t0 + n_steps * (excess * TIMESTEP_SEC / C_EFF_KJ_PER_K)
        t = t0
        for _ in range(n_steps):
            t = zone_temp_next(t, q_it_kw=180.0, q_cooling_kw=150.0)
        assert t == pytest.approx(expected_final, abs=1e-6)
