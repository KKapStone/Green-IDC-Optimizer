"""
칠러(Chiller) 전력 소비 모델

칠러는 냉각수를 냉각시키는 기계식 냉동기다.
외기 온도가 높을수록 칠러가 더 많은 전력을 소비한다.

핵심 지표: COP (Coefficient Of Performance, 성능계수)
  - COP = 제거한 열량 / 소비한 전력
  - COP가 높을수록 효율적 (같은 전력으로 더 많은 열을 제거)
  - 외기 온도가 낮을수록 COP 상승 → 전력 절감
"""

from dataclasses import dataclass
from core.config.enums import CoolingMode
from core.config.constants import (
    WET_BULB_FREE_THRESHOLD_C,
    WET_BULB_HYBRID_THRESHOLD_C,
    CHILLER_MAX_CAPACITY_KW,
    CHILLER_OPTIMAL_PLR,
    CHILLER_PLR_PENALTY,
)


def calculate_wet_bulb_c(temp_c: float, humidity_pct: float) -> float:
    """간단 근사식 기반 습구 온도 계산.

    공식: T_wb ≈ T - (1 - RH/100) × (T - 4)
    실제 Stull 공식의 단순화 버전. ±1°C 정확도.
    """
    return temp_c - (1.0 - humidity_pct / 100.0) * (temp_c - 4.0)


@dataclass
class ChillerResult:
    """칠러 계산 결과"""

    cop: float                  # 성능계수 (무차원)
    chiller_power_kw: float     # 칠러 전력 소비량 (kW)
    cooling_mode: CoolingMode   # 냉각 모드

def calculate_cop(outdoor_temp_c: float, supply_temp_c: float = 20.0) -> float:
    """
    외기 온도와 공급 온도에 따른 칠러 COP(성능계수)를 계산한다.

    공식: COP = max(2.0, 6.0 - 0.1 × (T_outdoor - 15) + 0.25 × (T_supply - 20))

    물리적 의미:
      - 외기 15°C, 공급 20°C(설계 기준점): COP = 6.0
      - 외기 1°C 오를수록 COP 0.1 감소 (열 배출 어려워짐)
      - 공급 온도 1°C 높일수록 COP 0.25 증가 (냉동 사이클 온도 리프트 감소)
      - 공급 온도 낮출수록 COP 감소 → PUE 상승 (에너지 페널티)

    Args:
        outdoor_temp_c: 외기 온도 (°C)
        supply_temp_c: CRAH 공급 온도 설정값 (°C), 기본값 20.0 (설계값)

    Returns:
        칠러 COP (무차원, 최솟값 2.0)
    """
    return max(2.0, 6.0 - 0.1 * (outdoor_temp_c - 15.0) + 0.25 * (supply_temp_c - 20.0))


def calculate_chiller_power_kw(
    cooling_load_kw: float,
    outdoor_temp_c: float,
    supply_temp_c: float = 20.0,
    outdoor_humidity_pct: float = 50.0,
) -> ChillerResult:
    """
    냉각 부하 + 외기 + 습도 + PLR 보정 기반 칠러 전력 계산.

    냉각 모드 결정 (습구 온도 기준 — 잠열 부하 반영):
      - 습구 < 13°C: Free Cooling
      - 13~20°C: Hybrid (선형 보간)
      - > 20°C: Chiller (기계식 전면 가동)

    PLR 보정: 칠러는 60% 부하에서 최고 효율, 이탈 시 효율 감소
      cop_actual = cop_base × (1 - 0.3 × ((PLR - 0.6) / 0.4)^2)

    Args:
        cooling_load_kw: 제거해야 할 열량 (kW)
        outdoor_temp_c: 외기 dry-bulb 온도 (°C)
        supply_temp_c: CRAH 공급 온도 (°C)
        outdoor_humidity_pct: 외기 상대 습도 (%, 기본 50)

    Returns:
        ChillerResult (COP, 칠러 전력, 냉각 모드)
    """
    if cooling_load_kw < 0:
        raise ValueError(f"냉각 부하는 0 이상이어야 합니다. 입력값: {cooling_load_kw}")

    # 습구 온도로 cooling mode 결정 (잠열 반영)
    wet_bulb_c = calculate_wet_bulb_c(outdoor_temp_c, outdoor_humidity_pct)

    # 기본 COP에 PLR 보정 적용
    cop_base = calculate_cop(outdoor_temp_c, supply_temp_c)
    plr = min(1.0, cooling_load_kw / CHILLER_MAX_CAPACITY_KW)
    plr_factor = max(0.5, 1.0 - CHILLER_PLR_PENALTY * ((plr - CHILLER_OPTIMAL_PLR) / 0.4) ** 2)
    cop = cop_base * plr_factor

    if wet_bulb_c < WET_BULB_FREE_THRESHOLD_C:
        mode = CoolingMode.FREE_COOLING
        chiller_power_kw = 0.0
    elif wet_bulb_c <= WET_BULB_HYBRID_THRESHOLD_C:
        chiller_fraction = (wet_bulb_c - WET_BULB_FREE_THRESHOLD_C) / (
            WET_BULB_HYBRID_THRESHOLD_C - WET_BULB_FREE_THRESHOLD_C
        )
        mode = CoolingMode.HYBRID
        chiller_power_kw = (cooling_load_kw * chiller_fraction) / cop
    else:
        mode = CoolingMode.CHILLER
        chiller_power_kw = cooling_load_kw / cop

    return ChillerResult(cop=cop, chiller_power_kw=chiller_power_kw, cooling_mode=mode)
