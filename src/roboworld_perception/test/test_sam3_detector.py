"""프롬프트 임베딩 캐시 — 모델 없이 검증한다.

캐시 로직만 스텁으로 실제 경로를 태운다 — 모델도 GPU 도 안 쓴다.
다만 sam3_detector 모듈 자체가 torch 를 top-level 로 임포트하므로,
torch 가 없는 환경(CI)에서는 아래 importorskip 으로 건너뛴다.
"""
import numpy as np
import pytest

# sam3_detector 는 torch 를 **top-level 로** 임포트한다(기본 인자에 torch.bfloat16
# 이 쓰인다). CI 는 의도적으로 torch 를 안 깔고 geometry·tracker·pipeline 만
# 돌리므로(.github/workflows 주석 참고) 여기서 건너뛴다. 모델·GPU 는 여전히
# 필요 없다 — 이 파일은 캐시 로직만 스텁으로 태운다.
pytest.importorskip("torch", reason="CI 는 torch 를 설치하지 않는다")

from roboworld_perception.sam3_detector import (  # noqa: E402
    TEXT_CACHE_MAX, Sam3Detector, _boxes_to_model)


class _Inputs(dict):
    """processor 출력 흉내 — `.to(device)` · `.input_ids` · `.get()` 만 쓴다."""
    input_ids = "ids"

    def to(self, device):
        return self


class _Pooled:
    pooler_output = "embeds"


class _StubProcessor:
    def __call__(self, text=None, return_tensors=None):
        return _Inputs(attention_mask="mask")


class _StubModel:
    def __init__(self):
        self.calls = 0

    def get_text_features(self, **kwargs):
        self.calls += 1
        return _Pooled()


def _detector():
    """__init__(모델 로드) 을 우회하고 캐시 경로에 필요한 것만 채운다."""
    d = object.__new__(Sam3Detector)
    d._text_cache = {}
    d.device = "cpu"
    d.processor = _StubProcessor()
    d.model = _StubModel()
    return d


def test_normal_prompt_set_never_recomputes():
    """정상 사용(고정 프롬프트 반복)에서는 딱 한 번씩만 계산한다.

    이 캐시의 존재 이유가 그것이다 — 2026-08-21 실측으로 파이프라인 평균
    99.3 → 89.6ms(9.8%). Isaac 실운용은 detect_interval=1 이라 모든 프레임이
    키프레임이고 이 이득을 가장 크게 받는다. 상한을 넣다가 이걸 깨면 안 된다.
    """
    d = _detector()
    prompts = ["black bag", "keyboard", "book", "gray notebook"]
    for _ in range(50):                      # 50 프레임 분량
        for p in prompts:
            d._text_inputs(p)
    assert d.model.calls == len(prompts)     # 프롬프트당 1회
    assert len(d._text_cache) == len(prompts)


def test_cache_is_bounded_under_runtime_prompt_churn():
    """키가 런타임 입력이라 상한이 없으면 원리적으로 무한히 자란다."""
    d = _detector()
    for i in range(TEXT_CACHE_MAX * 3):
        d._text_inputs(f"prompt {i}")
    assert len(d._text_cache) <= TEXT_CACHE_MAX


def test_eviction_keeps_the_newest():
    """축출은 삽입 순서 — 가장 오래된 것이 나가고 최근 것은 남는다."""
    d = _detector()
    for i in range(TEXT_CACHE_MAX + 5):
        d._text_inputs(f"p{i}")
    assert f"p{TEXT_CACHE_MAX + 4}" in d._text_cache   # 마지막은 남고
    assert "p0" not in d._text_cache                   # 처음 것은 나갔다


def test_boxes_to_model_matches_sam3_normalization_convention():
    """transformers processing_sam3.Sam3Processor 의 _normalize_coordinates
    (is_bounding_box=True) + box_xyxy_to_cxcywh 를 좌표만으로 재현한다 —
    x 는 W, y 는 H 로 나누고 xyxy → cxcywh."""
    # 640x480 프레임의 (64,96)-(192,288) 박스. cx=128/640=0.2, cy=192/480=0.4,
    # w=128/640=0.2, h=192/480=0.4.
    out = _boxes_to_model([[64, 96, 192, 288]], original_size=(480, 640))
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], [0.2, 0.4, 0.2, 0.4], atol=1e-6)


def test_boxes_to_model_handles_multiple_boxes_independently():
    boxes = [[0, 0, 640, 480], [64, 96, 192, 288]]
    out = _boxes_to_model(boxes, original_size=(480, 640))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], [0.5, 0.5, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [0.2, 0.4, 0.2, 0.4], atol=1e-6)
