"""PUE 물리 모델 단위 테스트 — 공식 정확성, 물리 불변식, 경계값."""

import pytest

from domain.thermodynamics.pue import calculate_pue, PUEResult
from core.config.constants import PUE_BENCHMARK


class TestPUEFormula:
    def test_pue_definition(self):
        # PUE = (IT + 냉각 + 기타) / IT
        result = calculate_pue(it_power_kw=100.0, cooling_power_kw=20.0, other_power_kw=5.0)
        assert result.pue == pytest.approx(125.0 / 100.0)

    def test_pue_always_at_least_one(self):
        # PUE < 1.0은 물리적으로 불가 (냉각 전력이 있으면 항상 > 1)
        result = calculate_pue(it_power_kw=100.0, cooling_power_kw=10.0, other_power_kw=0.0)
        assert result.pue >= 1.0

    def test_zero_cooling_and_other_gives_pue_one(self):
        # 냉각/기타 전력이 0이면 PUE = 1.0 (이상적 데이터센터)
        result = calculate_pue(it_power_kw=100.0, cooling_power_kw=0.0, other_power_kw=0.0)
        assert result.pue == pytest.approx(1.0)

    def test_other_power_defaults_to_5pct(self):
        # other_power_kw 미지정 시 IT 전력의 5%로 추정
        result = calculate_pue(it_power_kw=200.0, cooling_power_kw=0.0)
        assert result.other_power_kw == pytest.approx(10.0)  # 200 × 5%

    def test_total_power_is_sum(self):
        result = calculate_pue(it_power_kw=100.0, cooling_power_kw=15.0, other_power_kw=5.0)
        assert result.total_power_kw == pytest.approx(120.0)

    def test_pue_increases_with_cooling_power(self):
        pue_low = calculate_pue(100.0, 10.0, 5.0).pue
        pue_high = calculate_pue(100.0, 40.0, 5.0).pue
        assert pue_high > pue_low

    def test_pue_decreases_with_it_power(self):
        # IT 전력이 클수록 냉각 비율 줄어 PUE 개선
        pue_small_it = calculate_pue(50.0, 20.0, 2.5).pue
        pue_large_it = calculate_pue(200.0, 20.0, 10.0).pue
        assert pue_large_it < pue_small_it


class TestPUEBenchmark:
    def test_efficiency_vs_benchmark_reflects_naver(self):
        # PUE = NAVER 수준이면 비율 ≈ 1.0
        naver_pue = PUE_BENCHMARK["naver_chuncheon"]  # 1.09
        # PUE = 1.09 → other_power = IT × 5%, 역산: 1.09 = (IT + cooling + other) / IT
        # → cooling = (1.09 - 1.05) × IT = 0.04 × IT
        it_kw = 100.0
        cooling_kw = (naver_pue - 1.05) * it_kw
        result = calculate_pue(it_kw, cooling_kw)
        assert result.efficiency_vs_benchmark == pytest.approx(1.0, abs=0.01)

    def test_high_pue_has_efficiency_ratio_above_one(self):
        # PUE 2.0 (비효율) → benchmark 대비 비율 > 1
        result = calculate_pue(100.0, 90.0, 5.0)  # PUE ≈ 1.95
        assert result.efficiency_vs_benchmark > 1.0


class TestPUEValidation:
    def test_zero_it_power_raises(self):
        with pytest.raises(ValueError, match="0보다 커야"):
            calculate_pue(it_power_kw=0.0, cooling_power_kw=10.0)

    def test_negative_it_power_raises(self):
        with pytest.raises(ValueError, match="0보다 커야"):
            calculate_pue(it_power_kw=-50.0, cooling_power_kw=10.0)

    def test_negative_cooling_power_raises(self):
        with pytest.raises(ValueError, match="0 이상"):
            calculate_pue(it_power_kw=100.0, cooling_power_kw=-5.0)

    def test_result_is_pue_result_dataclass(self):
        result = calculate_pue(100.0, 20.0)
        assert isinstance(result, PUEResult)
        assert result.it_power_kw == pytest.approx(100.0)
        assert result.cooling_power_kw == pytest.approx(20.0)
