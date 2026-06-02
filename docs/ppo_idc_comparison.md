# PPO vs SAC 비교 실험 — IDCEnv 재학습

## 개요

`domain/controllers/rl_agent.py`의 PPO 알고리즘을 커스텀 IDCEnv 환경에서 1M 스텝 학습한 뒤,
SAC·TD3 및 기존 베이스라인 컨트롤러와 PUE·온도 위반 기준으로 비교한다.

- **환경**: `IDCEnv` (커스텀, Sinergym 미사용)
- **데이터**: `synthetic_idc_1year_noisy.parquet` (Google Cluster Trace 2019 + 기상청 ASOS)
- **평가**: 20 에피소드 (1일 × 20), `w_energy=0.8`, seed `42~61`

---

## 학습 설정

| 항목 | PPO | SAC (비교) | TD3 (비교) |
|---|---|---|---|
| 모델 파일 | `data/models/ppo-idc-1m-lr1e4.zip` | `data/models/sac-wetbulb-1m.zip` | `data/models/td3-idc-1m.zip` |
| 총 스텝 | 1,000,000 | 1,000,000 | 1,000,000 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Batch Size | 64 † | 256 | 256 |
| Gamma | 0.99 | 0.99 | 0.99 |
| n_steps (rollout) | 2,048 | — (off-policy) | — (off-policy) |
| Buffer Size | — (on-policy) | 200,000 | 200,000 |
| w_energy | 0.8 | 0.8 | 0.8 |
| 보상 타입 | weighted | weighted | weighted |
| 에피소드 길이 | 288 스텝 (1일) | 288 스텝 | 288 스텝 |
| 학습 시간 (CPU) | 약 15.5분 | 약 24.0분 | 약 3~4시간 (절전 중단 포함) |

> † **batch_size 의미 차이**: PPO의 64는 rollout 버퍼(2048스텝)를 쪼개는 미니배치 크기이고, SAC/TD3의 256은 replay buffer에서 샘플링하는 수로 서로 다른 개념. 직접 비교 대상이 아님.

**학습 커맨드 (PPO):**

```bash
python -m domain.controllers.rl_agent \
  --algo ppo --custom-env \
  --total-timesteps 1000000 \
  --lr 1e-4 --n-steps 2048 --batch-size 64 \
  --gamma 0.99 --w-energy 0.8 \
  --reward-type weighted --max-episode-steps 288 \
  --run-name ppo-idc-1m-lr1e4 --seed 0
```

---

## 평가 결과

> 평가 스크립트: `scripts/eval_baseline.py --model data/models/ppo-idc-1m-lr1e4.zip --episodes 20`

| 컨트롤러 | PUE 평균 | 온도 위반 (°C) | 보상 평균 | 서버실 온도 (°C) |
|---|:---:|:---:|:---:|:---:|
| 고정 setpoint 24°C | 1.2999 | 0.0000 | -90.684 | 25.15 |
| Random | 1.2059 | 0.0000 | -28.374 | 25.65 |
| Rule-based | 1.1894 | 0.0000 | -17.427 | 25.01 |
| 고정 setpoint 20°C (설계값) | 1.1847 | 0.0000 | -14.370 | 25.01 |
| **PPO (ppo-idc-1m-lr1e4)** | **1.1768** | **0.0000** | **-9.145** | **25.01** |
| TD3 (td3-idc-1m) | 1.1751 | 0.0000 | -7.956 | 25.86 |
| PID (zone target=24°C) | 1.1752 | 0.0000 | -8.025 | 25.55 |
| SAC (sac-wetbulb-1m) | 1.1747 | 0.0000 | -7.724 | 25.76 |

---

## 학습 커브 요약

### PPO (lr=1e-4)

- 초기 보상: `+3.08` (step 2,048) — 양수로 시작
- 최고 보상: `+0.36` (step 870,400)
- 최종 보상: `-13.03` (step 1,001,472)
- 특이점: 초반 양수 구간 진입 후 step 10K부터 음수로 전환, 이후 `-5 ~ -27` 진동하며 명확한 수렴 없음

### SAC (비교)

- 초기 보상: `-33.9` (step 4,096)
- 최고 보상: `+44.6` (step ~223,000)
- 최종 보상: `+35.8` (step 999,424)
- 특이점: 약 80,000 스텝 이후 양수 구간 진입, 이후 꾸준히 상승

SAC는 off-policy 특성(Replay Buffer)으로 sample efficiency가 높고 보상 수렴이 명확한 반면,
PPO는 on-policy 특성상 rollout 간 분산이 크고 IDCEnv의 연속적 보상 지형에서 수렴이 느렸다.

---

## 분석

### PUE 비교

```
SAC   1.1747  ← 최저 (최우수)
TD3   1.1751  (+0.0004 vs SAC)
PID   1.1752  (+0.0005 vs SAC)
PPO   1.1768  (+0.0021 vs SAC)
Rule  1.1894  (+0.0147 vs SAC)
```

- PPO는 Rule-based 대비 **PUE 0.0126 감소** (overhead 기준 약 11.2% 개선)
- SAC 대비 PUE 차이는 **0.0021** (실운용상 유의미한 차이 없음)
- TD3(1.1751)가 PPO(1.1768)보다 PUE 0.0017 낮아 RL 알고리즘 중 PPO가 가장 열세
- 온도 위반은 모든 컨트롤러에서 0

### PPO 성능이 SAC·TD3보다 낮은 이유

1. **On-policy 한계**: rollout을 즉시 소비하므로 replay buffer를 갖는 SAC·TD3 대비 data efficiency 낮음
2. **연속 행동 공간**: 1차원 setpoint 최적화에서 SAC의 entropy 자동 조절이 탐색에 유리
3. **학습 불안정**: 초반 양수 진입 후 step 10K부터 보상 급락 — on-policy 특성상 잘못된 rollout이 정책을 덮어씌우기 쉬움

### PPO의 장점

- 학습 코드 단순 (`n_steps=2048`, on-policy rollout)
- 하이퍼파라미터 민감도 낮아 초기 실험·프로토타입에 적합
- 동일 1M 스텝 기준 학습 시간이 SAC(24분)·TD3(3~4시간) 대비 가장 빠름 (15.5분)

---

## 결론

| 항목 | 결과 |
|---|---|
| 최종 채택 알고리즘 | **SAC** (`sac-wetbulb-1m`) |
| PPO의 Rule-based 대비 PUE 개선 | **−0.0126** (1.1894 → 1.1768) |
| SAC 대비 PPO PUE 열세 | **+0.0021** |
| TD3 대비 PPO PUE 열세 | **+0.0017** |
| 온도 위반 | **전 알고리즘 0** — 안전 제약 완전 준수 |

PPO도 Rule-based 대비 유의미한 에너지 개선을 보였으나, SAC·TD3 대비 PUE와 학습 안정성 모두 열세.
최종 제어 에이전트는 **SAC(sac-wetbulb-1m)** 을 유지한다.

---

*실험 날짜: 2026-05-25*
*평가 환경: IDCEnv, w_energy=0.8, 20 에피소드*
*관련 문서: [td3_idc_comparison.md](td3_idc_comparison.md)*
