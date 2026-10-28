"""Inference engine and prediction interface for NewsLens AI.

Provides structured category predictions, class probabilities, input validation,
and inference explanations.

Author: Daniel Richard Xavier
"""

import os
from typing import Dict, Any, Optional, List, Union
import numpy as np

from src.data_loader import CATEGORY_MAP_ID_TO_NAME
from src.preprocessing import clean_text, combine_headline_and_body, validate_input_text
from src.model import load_pipeline


class NewsLensPredictor:
    """Predictor class encapsulating model loading and single/batch inference."""

    def __init__(self, model_path: Optional[str] = None, pipeline: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize predictor from model path or existing pipeline instance."""
        if pipeline is not None:
            self.pipeline = pipeline
            self.metadata = metadata or {}
            self.model_path = model_path or "in-memory"
        elif model_path is not None:
            self.model_path = model_path
            self.pipeline, self.metadata = load_pipeline(model_path)
        else:
            raise ValueError("Either model_path or pipeline must be provided.")

        self.class_mapping = CATEGORY_MAP_ID_TO_NAME
        self.class_names = [self.class_mapping[i] for i in sorted(self.class_mapping.keys())]

    def predict(
        self,
        text: Optional[str] = None,
        headline: Optional[str] = None,
        body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Classify a news article based on headline and/or body.
        
        Args:
            text: Unified article text (if headline and body are not passed separately).
            headline: News headline / title.
            body: News description / body.

        Returns:
            Structured prediction dictionary.
        """
        # Determine raw input
        if headline is not None or body is not None:
            combined = combine_headline_and_body(headline, body)
        else:
            combined = text if text is not None else ""

        # Validate input
        is_valid, cleaned, error_msg = validate_input_text(combined)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "predicted_category": None,
                "predicted_category_id": None,
                "confidence": 0.0,
                "probabilities": {cname: 0.0 for cname in self.class_names},
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()) if cleaned else 0
            }

        # Inference
        try:
            pred_id = int(self.pipeline.predict([cleaned])[0])
            pred_category = self.class_mapping.get(pred_id, f"Unknown ({pred_id})")

            # Probability extraction if supported
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba([cleaned])[0]
                prob_dict = {
                    self.class_mapping[idx]: float(round(probs[idx], 4))
                    for idx in range(len(self.class_names))
                }
                confidence = float(round(probs[pred_id], 4))
            else:
                prob_dict = {cname: (1.0 if cname == pred_category else 0.0) for cname in self.class_names}
                confidence = 1.0

            result = {
                "success": True,
                "error": None,
                "predicted_category": pred_category,
                "predicted_category_id": pred_id,
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2),
                "probabilities": prob_dict,
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
                "model_name": self.metadata.get("model_name", "NewsLens Pipeline"),
                "disclaimer": "Predicted probabilities are model estimates based on training patterns. NewsLens AI classifies topic domains; it does not verify factual accuracy or detect misinformation."
            }
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Inference error: {str(e)}",
                "predicted_category": None,
                "predicted_category_id": None,
                "confidence": 0.0,
                "probabilities": {cname: 0.0 for cname in self.class_names},
                "cleaned_text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()) if cleaned else 0
            }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch inference over a list of texts."""
        return [self.predict(text=t) for t in texts]
