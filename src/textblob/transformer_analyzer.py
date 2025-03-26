"""
This file is for integrating a pre trained transformer model from Hugging Face.

"""
from typing import Dict, Any, Optional
from transformers import pipeline
from textblob.base import BaseSentimentAnalyzer

class TransformerAnalyzer(BaseSentimentAnalyzer):
    """
    A sentiment analyzer that uses a pre-trained Transformer model
    from Hugging Face for sentiment analysis.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        pipeline (Pipeline): The Hugging Face pipeline for sentiment analysis.
    """
    
    def __init__(self, model_name = 'distilbert-base-uncased-finetuned-sst-2-english'):

        super().__init__()
        self.model_name = model_name
        try:
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the pipeline with model '{self.model_name}': {e}")

    
    def analyze(self, text: str) -> Optional[Dict[str, Any]]:

        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")

        try:
            raw_result = self.pipeline(text)
            if isinstance(raw_result, list) and all(isinstance(item, dict) for item in raw_result):
                result = [item for item in raw_result if isinstance(item, dict)]
            else:
                raise TypeError("Unexpected result type from pipeline. Expected List[Dict[str, Any]].")
            if result:
                return result[0]
        except Exception as e:
            raise RuntimeError(f"Error during sentiment analysis: {e}")

        return None