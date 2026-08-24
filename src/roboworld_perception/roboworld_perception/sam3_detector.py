"""Text-prompted open-vocabulary detection/segmentation via SAM3 (transformers)."""
import numpy as np
import torch
from PIL import Image

# ponytail: SAM3 text encoder is English CLIP; map Korean prompts here.
PROMPT_ALIASES = {
    "물통": "thermos",  # 초록 보온병: "water bottle"은 0.3~0.45로 불안정, thermos는 ~0.9
    "마우스": "computer mouse",
    "필통": "pencil case",
    "노트북": "laptop",
    "책": "book",
    "스마트폰": "smartphone",
    "장갑": "glove",
    "천": "white cloth",
    "블록": "pink foam block",
}


# 프롬프트 임베딩 캐시 상한. 키가 런타임 입력(/perception/prompt 토픽)이라
# 상한이 없으면 원리적으로 무한히 자란다.
#
# **크기 자체는 작다** — 항목당 16.2 KB 다(계산: text_embeds [1, 32, 256] bf16
# = 16 KB + attention_mask [1, 32] int64 = 256 B. 토크나이저가 프롬프트 길이와
# 무관하게 32 로 고정 패딩하므로 상수다). 1,000 개를 쌓아도 16 MB, 12 GB 의
# 0.13% 라 이 상한은 OOM 대책이 아니라 **무한 증가라는 성질 자체를 닫는 가드**다.
#
# 64 는 실기 최대 프롬프트 수(test5 의 4 개)의 16 배다. 정상 사용에서는 절대
# 닿지 않으므로 축출 정책(FIFO/LRU)이 결과를 바꾸지 않는다 — 가장 단순한
# 삽입 순서를 쓴다. **상한에 닿는 것 자체가 비정상 신호**다.
TEXT_CACHE_MAX = 64


def parse_prompts(s):
    """쉼표 구분 프롬프트 문자열 → 리스트 (노드·스크립트 공용 규칙)."""
    return [p.strip() for p in s.split(",") if p.strip()]


class Sam3Detector:
    def __init__(self, model_id="facebook/sam3", device=None, threshold=0.4,
                 mask_threshold=0.5, dtype=torch.bfloat16, image_size=0,
                 compile_model=False):
        from transformers import Sam3Config, Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16이 fp32 대비 3.7배 빠르고 score 차이 없음 (실측 548ms -> 150ms)
        self.dtype = dtype if self.device == "cuda" else torch.float32
        kwargs = {}
        proc_kwargs = {}
        if image_size:  # 0 = 기본 1008px. 축소 시 속도↑, 작은 물체 검출력↓
            config = Sam3Config.from_pretrained(model_id)
            config.image_size = image_size
            kwargs["config"] = config
            proc_kwargs["size"] = {"height": image_size, "width": image_size}
        self.model = Sam3Model.from_pretrained(
            model_id, dtype=self.dtype, **kwargs).to(self.device).eval()
        if compile_model:
            self.model = torch.compile(self.model)
        self.processor = Sam3Processor.from_pretrained(model_id, **proc_kwargs)
        self.threshold = threshold
        # 연관(association)용 하한 — threshold 미만~이 값 이상의 저점수 검출도
        # 반환한다. 트래커가 기존 트랙 유지에만 쓰고 새 트랙은 못 만든다
        # (ByteTrack의 저점수 2차 매칭). 부분 가림 추적이 끊기지 않게 함.
        self.assoc_threshold = min(0.1, threshold)
        self.mask_threshold = mask_threshold
        self._warned = set()
        self._text_cache = {}  # 프롬프트별 텍스트 임베딩 (아래 _text_inputs 참고)

    def _text_inputs(self, text):
        """프롬프트의 텍스트 임베딩을 캐시해 돌려준다.

        토큰화만 캐시하면 CLIP 텍스트 인코더가 **키프레임마다 프롬프트마다**
        다시 돈다 (실측 29.7ms / 3프롬프트, 키프레임 SAM3 시간의 7%).
        프롬프트는 런 중 고정이므로 한 번만 계산하면 된다.

        forward 는 text_embeds 를 주면 get_text_features 를 건너뛴다
        (modeling_sam3.py 의 `if text_embeds is None:`). 여기서 하는 계산이
        그 분기가 하던 것과 같은 것이라 결과가 바뀌지 않는다 — 다만 XOR
        검사가 있어 input_ids 는 함께 넘기면 안 되고, attention_mask 는
        text_mask 로 따로 쓰이므로 계속 넘겨야 한다.
        """
        if text not in self._text_cache:
            if len(self._text_cache) >= TEXT_CACHE_MAX:
                # 삽입 순서가 가장 오래된 항목. 위 TEXT_CACHE_MAX 주석 참고 —
                # 여기 닿았다면 프롬프트가 런타임에 계속 바뀌고 있다는 뜻이다.
                self._text_cache.pop(next(iter(self._text_cache)))
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            self._text_cache[text] = {
                "text_embeds": self.model.get_text_features(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    return_dict=True).pooler_output,
                "attention_mask": inputs.get("attention_mask"),
            }
        return self._text_cache[text]

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, prompts: list[str]) -> list[dict]:
        """rgb: HxWx3 uint8. Returns [{label, mask(HxW bool), box(xyxy), score}].

        Vision embedding is computed once and reused across prompts.
        """
        image = Image.fromarray(rgb)
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        vision_embeds = self.model.get_vision_features(
            pixel_values=img_inputs.pixel_values.to(self.dtype))
        target_sizes = img_inputs.get("original_sizes").tolist()

        detections = []
        for label in prompts:
            text = PROMPT_ALIASES.get(label, label)
            if not text.isascii() and label not in self._warned:
                self._warned.add(label)
                print(f"[경고] '{label}'은 별칭 테이블에 없는 한글 프롬프트입니다. "
                      f"SAM3는 영어 기반이라 검출이 안 될 수 있습니다 — "
                      f"영어로 입력하거나 PROMPT_ALIASES에 추가하세요.", flush=True)
            outputs = self.model(vision_embeds=vision_embeds,
                                 **self._text_inputs(text))
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.assoc_threshold,
                mask_threshold=self.mask_threshold, target_sizes=target_sizes)[0]
            detections.extend({
                "label": label,
                "mask": mask.cpu().numpy().astype(bool),
                "box": box.float().cpu().numpy().astype(float),
                "score": float(score),
            } for mask, box, score in zip(results["masks"], results["boxes"],
                                          results["scores"]))
        return detections
