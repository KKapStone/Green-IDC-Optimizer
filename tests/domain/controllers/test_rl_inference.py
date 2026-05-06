"""RLInference 단위 테스트.

호환 모델이 없을 때는 predict 테스트를 skip 하고, 로딩 검증/싱글톤만 검증한다.
"""

from pathlib import Path

import numpy as np
import pytest

from domain.controllers.rl_inference import RLInference


@pytest.fixture(autouse=True)
def reset_singleton():
    """클래스 변수 _instance / _loaded_path 를 매 테스트마다 초기화."""
    RLInference._instance = None
    RLInference._loaded_path = None
    yield
    RLInference._instance = None
    RLInference._loaded_path = None


class TestFileChecks:
    def test_missing_zip_raises(self, tmp_path):
        missing = tmp_path / "no_such_model.zip"
        with pytest.raises(FileNotFoundError, match="RL 모델 없음"):
            RLInference(str(missing))

    def test_missing_vecnorm_raises(self, tmp_path):
        # zip 만 있고 _vecnorm.pkl 이 없을 때
        zip_path = tmp_path / "fake.zip"
        zip_path.write_bytes(b"\x00")  # 더미 zip
        with pytest.raises(FileNotFoundError, match="VecNormalize 통계 없음"):
            RLInference(str(zip_path))


def _find_compatible_model() -> Path | None:
    """models/ 하위에서 zip + _vecnorm.pkl 페어가 있는 모델 경로 반환. 없으면 None."""
    models_root = Path(__file__).resolve().parents[3] / "models"
    if not models_root.exists():
        return None
    for zip_path in models_root.rglob("*.zip"):
        stats = Path(str(zip_path).replace(".zip", "") + "_vecnorm.pkl")
        if stats.exists():
            return zip_path
    return None


@pytest.fixture(scope="module")
def compatible_model_path() -> Path:
    path = _find_compatible_model()
    if path is None:
        pytest.skip("호환 모델 없음 (zip + _vecnorm.pkl 페어 부재)")
    return path


class TestSingleton:
    def test_get_caches_instance_for_same_path(self, monkeypatch, compatible_model_path):
        from core.config.settings import settings as live_settings
        monkeypatch.setattr(live_settings, "rl_model_path", str(compatible_model_path))

        first = RLInference.get()
        second = RLInference.get()
        assert first is second
        assert RLInference._loaded_path == str(compatible_model_path)

    def test_get_reloads_when_path_changes(
        self, monkeypatch, compatible_model_path, tmp_path
    ):
        from core.config.settings import settings as live_settings
        monkeypatch.setattr(live_settings, "rl_model_path", str(compatible_model_path))
        first = RLInference.get()

        # 다른 (없는) 경로로 변경 → 재로드 시도 → FileNotFoundError
        bogus = tmp_path / "other.zip"
        monkeypatch.setattr(live_settings, "rl_model_path", str(bogus))
        with pytest.raises(FileNotFoundError):
            RLInference.get()
        # 재로드 시도 자체는 일어났는지 확인 (캐시된 인스턴스가 그대로 반환되지 않음)
        assert first is not None


class TestPredict:
    def test_predict_output_in_action_range(self, compatible_model_path):
        from domain.controllers.idc_env import T_SUPPLY_MAX, T_SUPPLY_MIN

        inf = RLInference(str(compatible_model_path))
        # IDCEnv obs (8,) 범위 안에서 임의값
        obs = np.array(
            [12.0, 25.0, 0.5, 60.0, 0.5, 25.0, 20.0, 200.0], dtype=np.float32
        )
        action = inf.predict(obs)
        assert isinstance(action, float)
        assert T_SUPPLY_MIN <= action <= T_SUPPLY_MAX
