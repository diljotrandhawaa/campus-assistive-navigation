"""Visual appearance features for detected object crops."""

import math
import os

import numpy as np
from PIL import Image


class AppearanceEncoder:
    def __init__(self, device):
        device = str(device)
        self.device = os.environ.get(
            "APPEARANCE_DEVICE",
            f"cuda:{device}" if device.isdecimal() else device,
        )
        self.model_id = os.environ.get(
            "APPEARANCE_MODEL", "facebook/dinov2-small",
        )

    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            use_safetensors=True,
        ).to(self.device).eval()

    def encode(self, image_bgr, detections, target=None):
        """Return {detection_index: normalized_feature_list}.

        Only the current target class needs appearance features.
        All other detections remain available to the existing UI and OCR.
        """
        import torch

        height, width = image_bgr.shape[:2]
        image = Image.fromarray(image_bgr[:, :, ::-1].copy())
        crops = []
        indices = []

        for index, detection in enumerate(detections):
            if target is not None and detection["label"] != target:
                continue

            box = detection["box"]
            x1 = max(0, min(width, math.floor(box[0] * width)))
            y1 = max(0, min(height, math.floor(box[1] * height)))
            x2 = max(0, min(width, math.ceil(box[2] * width)))
            y2 = max(0, min(height, math.ceil(box[3] * height)))

            # Very small crops are poor identity references.
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crop.thumbnail((224, 224), Image.Resampling.BICUBIC)

            # Preserve the complete object's proportions without center-cropping.
            square = Image.new("RGB", (224, 224), (127, 127, 127))
            square.paste(
                crop,
                ((224 - crop.width) // 2, (224 - crop.height) // 2),
            )

            indices.append(index)
            crops.append(square)

        features = {}

        with torch.inference_mode():
            for start in range(0, len(crops), 8):
                batch = crops[start:start + 8]
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                    do_resize=False,
                    do_center_crop=False,
                ).to(self.device)

                output = self.model(**inputs)
                vectors = output.last_hidden_state[:, 0].float()
                vectors = torch.nn.functional.normalize(
                    vectors, dim=-1, eps=1e-12,
                )
                vectors = vectors.cpu().numpy()

                for index, vector in zip(indices[start:start + 8], vectors):
                    if np.isfinite(vector).all():
                        features[index] = vector.tolist()

        return features