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
        self._text_cache = {}  # 프롬프트별 토큰화 결과 (키프레임마다 재계산 방지)

    def _text_inputs(self, text):
        if text not in self._text_cache:
            self._text_cache[text] = self.processor(
                text=text, return_tensors="pt").to(self.device)
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
