"""IT 전력 모델 단위 테스트 — SPECpower_ssj2008 공식 검증."""

import pytest

from domain.thermodynamics.it_power import (
    calculate_server_power_w,
    calculate_total_it_power_kw,
    ServerType,
    ServerSpec,
    CPU_SERVER,
    GPU_SERVER,
)
from core.config.constants import (
    CPU_SERVER_P_IDLE_W,
    CPU_SERVER_P_MAX_W,
    GPU_SERVER_P_IDLE_W,
    GPU_SERVER_P_MAX_W,
)


class TestServerPower:
    def test_idle_utilization_returns_p_idle(self):
        # CPU 사용률 0% → P_idle
        power = calculate_server_power_w(0.0, ServerType.CPU)
        assert power == pytest.approx(CPU_SERVER_P_IDLE_W)

    def test_full_utilization_returns_p_max(self):
        # CPU 사용률 100% → P_max
        power = calculate_server_power_w(1.0, ServerType.CPU)
        assert power == pytest.approx(CPU_SERVER_P_MAX_W)

    def test_linear_interpolation(self):
        # P = P_idle + (P_max - P_idle) × util
        util = 0.5
        expected = CPU_SERVER_P_IDLE_W + (CPU_SERVER_P_MAX_W - CPU_SERVER_P_IDLE_W) * util
        assert calculate_server_power_w(util, ServerType.CPU) == pytest.approx(expected)

    def test_gpu_server_higher_idle(self):
        # GPU 서버는 유휴에서도 CPU보다 전력 높음
        gpu_idle = calculate_server_power_w(0.0, ServerType.GPU)
        cpu_idle = calculate_server_power_w(0.0, ServerType.CPU)
        assert gpu_idle > cpu_idle

    def test_gpu_server_p_max(self):
        gpu_max = calculate_server_power_w(1.0, ServerType.GPU)
        assert gpu_max == pytest.approx(GPU_SERVER_P_MAX_W)

    def test_invalid_utilization_below_zero_raises(self):
        with pytest.raises(ValueError, match="0.0~1.0"):
            calculate_server_power_w(-0.1, ServerType.CPU)

    def test_invalid_utilization_above_one_raises(self):
        with pytest.raises(ValueError, match="0.0~1.0"):
            calculate_server_power_w(1.1, ServerType.CPU)

    def test_custom_spec(self):
        spec = ServerSpec(p_idle_w=100.0, p_max_w=400.0)
        power = calculate_server_power_w(0.5, custom_spec=spec)
        assert power == pytest.approx(100.0 + (400.0 - 100.0) * 0.5)

    @pytest.mark.parametrize("util", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_power_between_idle_and_max(self, util):
        power = calculate_server_power_w(util, ServerType.CPU)
        assert CPU_SERVER_P_IDLE_W <= power <= CPU_SERVER_P_MAX_W


class TestTotalITPower:
    def test_result_in_kilowatts(self):
        # 단위 확인: 400 CPU + 20 GPU, 유휴 → kW 단위
        power_kw = calculate_total_it_power_kw(0.0, 400, 20)
        cpu_idle_kw = CPU_SERVER_P_IDLE_W * 400 / 1000
        gpu_idle_kw = GPU_SERVER_P_IDLE_W * 20 / 1000
        assert power_kw == pytest.approx(cpu_idle_kw + gpu_idle_kw)

    def test_full_load_power(self):
        power_kw = calculate_total_it_power_kw(1.0, 400, 20)
        expected_kw = (CPU_SERVER_P_MAX_W * 400 + GPU_SERVER_P_MAX_W * 20) / 1000
        assert power_kw == pytest.approx(expected_kw)

    def test_power_increases_with_utilization(self):
        low = calculate_total_it_power_kw(0.2, 400, 20)
        high = calculate_total_it_power_kw(0.8, 400, 20)
        assert high > low

    def test_zero_servers_gives_zero(self):
        power = calculate_total_it_power_kw(0.5, 0, 0)
        assert power == pytest.approx(0.0)

    def test_design_config_in_reasonable_range(self):
        # 명세서 기준: 400 CPU + 20 GPU, 평균 60% 사용률 → 약 130~250 kW 예상
        power_kw = calculate_total_it_power_kw(0.6, 400, 20)
        assert 100.0 <= power_kw <= 300.0
