from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


class MultiModalEmbedder:
    def __init__(self, text_model_name: str, clip_model_name: str):
        self.text_model = SentenceTransformer(text_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model.to(self.device)
        self.clip_model.eval()

    def encode_text(self, items: List[str]) -> np.ndarray:
        if not items:
            return np.array([], dtype='float32')
        vectors = self.text_model.encode(items, normalize_embeddings=True)
        return np.asarray(vectors, dtype='float32')

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(features, p=2, dim=1)

    def _image_features_from_output(self, model_output) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, 'image_embeds') and model_output.image_embeds is not None:
            return model_output.image_embeds
        if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
            return model_output.pooler_output
        if hasattr(model_output, 'last_hidden_state') and model_output.last_hidden_state is not None:
            return model_output.last_hidden_state[:, 0, :]
        raise ValueError('Could not read image features from CLIP output')

    def _text_features_from_output(self, model_output) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, 'text_embeds') and model_output.text_embeds is not None:
            return model_output.text_embeds
        if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
            return model_output.pooler_output
        if hasattr(model_output, 'last_hidden_state') and model_output.last_hidden_state is not None:
            return model_output.last_hidden_state[:, 0, :]
        raise ValueError('Could not read text features from CLIP output')

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.array([], dtype='float32')
        with torch.no_grad():
            model_inputs = self.clip_processor(images=images, return_tensors='pt', padding=True)
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

            if hasattr(self.clip_model, 'get_image_features'):
                raw_features = self.clip_model.get_image_features(**model_inputs)
            else:
                vision_output = self.clip_model.vision_model(pixel_values=model_inputs['pixel_values'])
                pooled_output = self._image_features_from_output(vision_output)
                raw_features = self.clip_model.visual_projection(pooled_output)

            features = self._image_features_from_output(raw_features)
            features = self._normalize(features)
        return features.cpu().numpy().astype('float32')

    def encode_query_for_images(self, query: str) -> np.ndarray:
        with torch.no_grad():
            model_inputs = self.clip_processor(text=[query], return_tensors='pt', padding=True)
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

            if hasattr(self.clip_model, 'get_text_features'):
                raw_features = self.clip_model.get_text_features(**model_inputs)
            else:
                text_output = self.clip_model.text_model(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs.get('attention_mask'),
                )
                pooled_output = self._text_features_from_output(text_output)
                raw_features = self.clip_model.text_projection(pooled_output)

            features = self._text_features_from_output(raw_features)
            features = self._normalize(features)
        return features.cpu().numpy().astype('float32')
