"""Text-prompted open-vocabulary detection/segmentation via SAM3 (transformers)."""
import numpy as np
import torch
from PIL import Image

# ponytail: SAM3 text encoder is English CLIP; map Korean prompts here.
PROMPT_ALIASES = {
    "물통": "water bottle",
    "마우스": "computer mouse",
    "필통": "pencil case",
}


class Sam3Detector:
    def __init__(self, model_id="facebook/sam3", device=None, threshold=0.4,
                 mask_threshold=0.5):
        from transformers import Sam3Model, Sam3Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold
        self.mask_threshold = mask_threshold

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
            for mask, box, score in zip(results["masks"], results["boxes"],
                                        results["scores"]):
                detections.append({
                    "label": label,
                    "mask": mask.cpu().numpy().astype(bool),
                    "box": box.cpu().numpy().astype(float),
                    "score": float(score),
                })
        return detections
