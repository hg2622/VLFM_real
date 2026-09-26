# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from groundingdino.util.inference import load_model, predict
except ModuleNotFoundError:
    print("Could not import groundingdino. This is OK if you are only using the client.")

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


def _get_safe_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"[GroundingDINO] CUDA unavailable at runtime, falling back to CPU: {e}")
    return torch.device("cpu")


def _resolve_config_path(config_path: str) -> str:
    if os.path.exists(config_path):
        return config_path
    try:
        import groundingdino

        pkg_cfg = os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py")
        if os.path.exists(pkg_cfg):
            return pkg_cfg
    except Exception:
        pass
    return config_path


def _patch_transformers_bert_compat() -> None:
    """Compatibility shim for GroundingDINO with newer transformers builds."""
    try:
        from transformers.models.bert.modeling_bert import BertModel

        if not hasattr(BertModel, "get_head_mask"):
            def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            BertModel.get_head_mask = _get_head_mask  # type: ignore[attr-defined]
            print("[GroundingDINO] Applied transformers BertModel compatibility shim.")
    except Exception as e:
        print(f"[GroundingDINO] Could not apply transformers compatibility shim: {e}")


class GroundingDINO:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = _get_safe_device()
        _patch_transformers_bert_compat()
        config_path = _resolve_config_path(config_path)
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path).to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Convert image to tensor and normalize from 0-255 to 0-1
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # Remove detections whose class names do not exactly match the provided classes
        classes = caption_to_use[: -len(" .")].split(" . ")
        detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image, caption=payload["caption"]).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
