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


class Sam3Detector:
    def __init__(self, model_id="facebook/sam3", device=None, threshold=0.4,
                 mask_threshold=0.5, max_per_prompt=1):
        from transformers import Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        # 프롬프트당 유지할 인스턴스 수. open-vocab은 일치하는 모든 인스턴스를
        # 찾으므로, 기본은 최고 score 1개만 남긴다. 0 = 제한 없음.
        self.max_per_prompt = max_per_prompt

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, prompts: list[str]) -> list[dict]:
        """rgb: HxWx3 uint8. Returns [{label, mask(HxW bool), box(xyxy), score}].

        Vision embedding is computed once and reused across prompts.
        """
        image = Image.fromarray(rgb)
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        vision_embeds = self.model.get_vision_features(
            pixel_values=img_inputs.pixel_values)
        target_sizes = img_inputs.get("original_sizes").tolist()

        detections = []
        for label in prompts:
            text = PROMPT_ALIASES.get(label, label)
            text_inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold,
                mask_threshold=self.mask_threshold, target_sizes=target_sizes)[0]
            found = [{
                "label": label,
                "mask": mask.cpu().numpy().astype(bool),
                "box": box.cpu().numpy().astype(float),
                "score": float(score),
            } for mask, box, score in zip(results["masks"], results["boxes"],
                                          results["scores"])]
            found.sort(key=lambda d: -d["score"])
            if self.max_per_prompt > 0:
                found = found[:self.max_per_prompt]
            detections.extend(found)
        return detections
