# Runbook

## 개발환경 구성

1. **저장소 클론 및 폴더 이동**

   ```bash
   git clone <레포지토리_URL>
   cd AI-Green-IDC
   ```

2. **환경변수 파일 생성 및 설정**

   ```bash
   cp .env.example .env
   ```

   > `.env` 파일을 열어 `KMA_API_KEY` 등 각자의 로컬 환경에 맞는 값을 기입합니다.

3. **가상환경 생성 및 패키지 설치 (`uv` 사용)**
   ```bash
   uv venv
   source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
   uv sync
   ```

---

## Docker 실행

### 전체 실행

```bash
# 전체 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build

# 종료
docker compose down

# 로그 확인
docker compose logs -f api
docker compose logs -f forecast-service
docker compose logs -f control-service
docker compose logs -f simulation-service
docker compose logs -f dashboard
```

### simulation-service (Sinergym)

```bash
docker compose up --build simulation-service
```

### forecast-service

1. LGBM 모델 생성

   ```bash
   uv run python -m domain.forecasting.train.train_lgbm_it_load
   ```

   > `data/models/`에 joblib 파일이 생성됩니다.

2. 서비스 실행

   ```bash
   docker compose up --build forecast-service
   ```

3. 요청 예시

   <details>
   <summary>예시 Request</summary>

   ```bash
   # health check
   curl http://localhost:8001/health

   # LGBM IT_load 예측
   curl -X POST http://localhost:8001/api/v1/forecast \
      -H "Content-Type: application/json" \
      -d '{
         "prediction_target": "it_load",
         "model_type": "lgbm",
         "forecast_horizon_hours": 24,
         "include_prediction_interval": true
      }'
   ```

   </details>

---

## 로컬환경에서 서비스 실행

```bash
# API Gateway
uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

# Forecast Service
uv run --extra forecast uvicorn apps.forecast_service.main:app --reload --host 0.0.0.0 --port 8001

# Control Service
uv run --extra control uvicorn apps.control_service.main:app --reload --host 0.0.0.0 --port 8002

# Simulation Service
uv run uvicorn apps.simulation_service.main:app --reload --host 0.0.0.0 --port 8003

# Dashboard
uv run --extra dashboard streamlit run apps/dashboard/app.py --server.port 8501
```

---

## RL 학습 및 평가

커스텀 IDCEnv 기반 (Sinergym 불필요, 로컬에서 실행 가능).

> 백그라운드 실행 시 로그 디렉토리가 없으면 먼저 만드세요: `mkdir -p logs`

### 학습 (PPO)

```bash
# 포그라운드 (병렬 환경 8개)
uv run python -m domain.controllers.rl_agent \
  --algo ppo --custom-env --n-envs 8 \
  --total-timesteps 300000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.5 \
  --gamma 0.9 \
  --run-name ppo-exp-300k
```

```bash
# 백그라운드 (로그 파일로 저장)
nohup uv run python -m domain.controllers.rl_agent \
  --algo ppo --custom-env --n-envs 8 \
  --total-timesteps 300000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.5 \
  --run-name ppo-exp-300k \
  > logs/ppo-exp-300k.log 2>&1 &
```

### 학습 (SAC) ← **권장**

#### 권장 설정 (현재 best — PUE 1.1575)

```bash
nohup uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --total-timesteps 1000000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.8 \
  --gamma 0.99 \
  --lr 1e-4 \
  --batch-size 256 \
  --run-name sac-2pct-1m \
  > logs/sac-2pct-1m.log 2>&1 &
```

> 튜닝 근거:
> - `gamma 0.99`: 1일(288스텝) 에피소드 미래 보상 반영
> - `lr 1e-4`: gradient 진동 감소
> - `batch-size 256`: gradient 노이즈 감소
> - `w-energy 0.8`: PUE 신호 강조
> - `policy_kwargs={"net_arch": [256, 256, 128]}`: 네트워크 표현력 ↑ (rl_agent.py 코드)
> - `VecNormalize(norm_reward=True)`: SAC 학습 안정성 향상 (rl_agent.py 코드)

#### 빠른 sanity check (약 30초~1분)

```bash
uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --total-timesteps 30000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.8 --gamma 0.99 --lr 1e-4 --batch-size 256 \
  --run-name sac-sanity-30k
```

#### 변형 실험 (추가 탐색)

```bash
# hierarchical 보상 (안전/효율 분리)
nohup uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --total-timesteps 500000 \
  --max-episode-steps 288 \
  --reward-type hierarchical \
  --gamma 0.99 --lr 1e-4 --batch-size 256 \
  --run-name sac-hier-500k \
  > logs/sac-hier-500k.log 2>&1 &
```

### 진행 모니터링

```bash
# stdout 실시간 로그
tail -f logs/sac-affinity-500k.log

# rollout 단위 csv (timestep, ep_rew_mean, ep_len_mean)
tail -f data/logs/sac-affinity-500k.csv

# 학습 프로세스 확인
ps aux | grep rl_agent | grep -v grep
```

### 이어서 학습 (resume)

`--resume`은 모델(.zip)과 VecNormalize 통계(_vecnorm.pkl)를 함께 자동 로드합니다.

```bash
uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --resume data/models/sac-affinity-500k.zip \
  --total-timesteps 200000 \
  --max-episode-steps 288 \
  --reward-type hierarchical \
  --run-name sac-affinity-700k
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--algo` | ppo | 알고리즘 (`ppo` \| `sac`) |
| `--custom-env` | False | IDCEnv 사용 (커스텀 열역학 모델) |
| `--n-envs` | 1 | 병렬 환경 수 (>1이면 SubprocVecEnv) |
| `--reward-type` | weighted | 보상 함수 (`weighted` \| `hierarchical`) |
| `--total-timesteps` | 50000 | 총 학습 step |
| `--max-episode-steps` | 96 | 에피소드 길이 (288 = 1일, 5분 간격) |
| `--run-name` | ppo-baseline | 모델 저장 이름 |
| `--w-energy` | 0.5 | weighted 보상 에너지 항 가중치 (0~1) |
| `--gamma` | 0.9 | 할인율 |
| `--lr` | 3e-4 | learning rate |
| `--n-steps` | 2048 | PPO rollout buffer 길이 |
| `--batch-size` | 64 | 미니배치 크기 |
| `--ent-coef` | 0.0 | PPO entropy 보너스 계수 (탐색 강도) |
| `--log-std-init` | 0.0 | PPO 초기 action std (log scale) |
| `--device` | auto | `auto` \| `cuda` \| `cpu` |
| `--resume` | None | 이어서 학습할 모델 경로 (.zip) |

#### 보상 함수 선택

- **`weighted`** (권장): 안전 우선 + PUE 선형 신호
  - 위반 시: `-temp_violation × 1.5` (w_energy 무관, 절대 위반 금지)
  - 안전 시: `-w_energy × pue_overhead + (0.25 - pue_overhead) × 1.5`
    - NAVER 1.20 기준 선형 신호, 1.20 이상에서도 음수 영역 유지 (학습 신호 평탄 방지)
- **`hierarchical`**: 위반 시 `-temp_violation`, 안전 시 `-pue_overhead` — 안전/효율 목표 분리

#### 병렬 환경 가이드

- PPO: `--n-envs 8` 권장 (on-policy, 병렬에서 큰 이득)
- SAC: `--n-envs 4` 권장 (off-policy, 병렬 효율 낮지만 sample 수집은 도움)
- 하드웨어 코어 수 확인: `sysctl -n hw.ncpu` (macOS) / `nproc` (Linux)

학습 로그는 `data/logs/{run-name}.csv`에, 모델은 `data/models/{run-name}.zip` + `_vecnorm.pkl`에 자동 저장됩니다.

### Docker로 학습

```bash
# 빌드
docker compose build rl-service
```

```bash
# 학습 실행 (포그라운드)
docker compose run --rm rl-service uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --total-timesteps 500000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.8 --gamma 0.99 --lr 1e-4 --batch-size 256 \
  --run-name sac-tuned-w08-500k
```

```bash
# 학습 실행 (백그라운드)
nohup docker compose run --rm rl-service uv run python -m domain.controllers.rl_agent \
  --algo sac --custom-env --n-envs 4 \
  --total-timesteps 500000 \
  --max-episode-steps 288 \
  --reward-type weighted \
  --w-energy 0.8 --gamma 0.99 --lr 1e-4 --batch-size 256 \
  --run-name sac-tuned-w08-500k \
  > logs/sac-tuned-w08-500k.log 2>&1 &
```

> `rl-service`는 `profiles: [rl]`로 설정되어 있어 `docker compose up --build` 시 자동 포함되지 않습니다.

### 평가

```bash
PYTHONPATH=. uv run python scripts/eval_baseline.py --model data/models/sac-2pct-1m.zip
PYTHONPATH=. uv run python scripts/check_action.py --model data/models/sac-2pct-1m.zip
```

### 실험 결과 비교

> **주의**: 환경(IDCEnv)이 실측 데이터 + 습도/PLR 칠러 + 모드별 fan 모델로 정밀화됨. 이전 환경 모델은 호환 불가.

| 모델 | PUE | ep_rew_mean | 온도 위반 | 비고 |
|---|---|---|---|---|
| Rule-based | 1.1905 | -18.207 | 0.000 | 베이스라인 (현실 환경) |
| 고정 setpoint 20°C | 1.1847 | -14.370 | 0.000 | 설계값 |
| 고정 setpoint 24°C | 1.2999 | -90.684 | 0.000 | 너무 높음 |
| sac-norm-500k | 1.1785 | -31.833 | 0.000 | norm_reward 적용 |
| **sac-2pct-1m** ← **best** | **1.1755** | **-8.240** | **0.000** | 1M + 보상 임계값 0.25 + 큰 네트워크 |

- Rule-based 대비 **PUE 1.26% 개선** (산업 표준 1~3% 범위)
- 고정 24°C 대비 **9.6% 개선**
- NAVER 춘천(1.09)까지 격차 0.0855
- **위반 0** 안전 보장, **Setpoint 18~25°C 동적 조정**

> 이전 합성 환경의 11.5% 개선은 환경 단순성에서 오는 sim-to-real gap 큰 수치였음. 현재 1.26%는 실측 데이터 + 산업 표준 물리 기반 신뢰 가능한 결과.
