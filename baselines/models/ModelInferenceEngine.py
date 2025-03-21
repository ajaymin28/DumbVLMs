from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

class ModelInferenceEngine(ABC):
    """Abstract base class for model inference engines used in visual LLM evaluation"""
    
    def __init__(self, 
                 model: Any,
                 tokenizer: Any,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 device: str = "auto"):
        """
        Initialize the inference engine with model parameters.
        
        Args:
            model: The underlying model instance
            tokenizer: The tokenizer instance
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens for generation (default: 512)
            device: Device to run inference on (default: "auto")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metrics tracking
        self.inference_count = 0
        self.total_latency = 0.0
        
    @abstractmethod
    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set or update the tokenizer for the model"""
        pass
    
    @abstractmethod
    def get_response(self, data: Union[str, Dict, List]) -> Dict:
        """
        Generate a response from the model given input data
        
        Args:
            data: Input data (text, dict with image/text, or list of inputs)
            
        Returns:
            Dict containing response and metadata
        """
        pass
    
    @abstractmethod
    def predict(self, data: Union[str, Dict, List]) -> Any:
        """
        Make a prediction based on input data
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction output (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def preprocess_input(self, data: Any) -> Any:
        """
        Preprocess input data before inference
        
        Args:
            data: Raw input data
            
        Returns:
            Processed data ready for model input
        """
        pass
    
    @abstractmethod
    def postprocess_output(self, output: Any) -> Dict:
        """
        Postprocess model output
        
        Args:
            output: Raw model output
            
        Returns:
            Processed output in dictionary format
        """
        pass
    
    def evaluate_performance(self, test_data: List) -> Dict:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_data: List of test samples
            
        Returns:
            Dict containing performance metrics
        """
        metrics = {
            "avg_latency": 0.0,
            "success_rate": 0.0,
            "total_samples": len(test_data)
        }
        return metrics
    
    def batch_predict(self, 
                     data_batch: List,
                     batch_size: int = 32) -> List:
        """
        Process a batch of inputs
        
        Args:
            data_batch: List of input data
            batch_size: Size of each batch
            
        Returns:
            List of predictions
        """
        predictions = []
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            batch_preds = [self.predict(item) for item in batch]
            predictions.extend(batch_preds)
        return predictions
    
    def get_model_info(self) -> Dict:
        """
        Return information about the model configuration
        
        Returns:
            Dict containing model parameters and configuration
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "inference_count": self.inference_count
        }
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data format and content
        
        Args:
            data: Input data to validate
            
        Returns:
            Boolean indicating if input is valid
        """
        if data is None:
            self.logger.error("Input data cannot be None")
            return False
        return True
    
    def save_state(self, path: str) -> None:
        """
        Save model state to disk
        
        Args:
            path: File path to save state
        """
        self.logger.warning("Save state not implemented for base class")