"""
Inference utilities for bridging FastAPI and ML pipeline.

Provides clean interface for loading models and running predictions
in API context.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import torch
from pathlib import Path


class InferencePipeline:
    """
    End-to-end inference pipeline for hypoglycemia prediction.
    
    Bridges the gap between API requests and the core ML pipeline.
    Handles model loading, data preprocessing, and prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model file (keras, pytorch, etc.)
            device: Device for PyTorch models ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.model_type = None  # 'keras', 'pytorch', etc.
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file.
        
        Automatically detects model type from file extension.
        
        Args:
            model_path: Path to model file
        """
        self.model_path = model_path
        path = Path(model_path)
        
        if path.suffix in ['.h5', '.keras']:
            # TensorFlow/Keras model
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            self.model_type = 'keras'
        elif path.suffix in ['.pt', '.pth']:
            # PyTorch model
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_type = 'pytorch'
        else:
            raise ValueError(
                f"Unsupported model file type: {path.suffix}. "
                f"Supported types: .h5, .keras, .pt, .pth"
            )
    
    def preprocess(self, data: Union[np.ndarray, pd.DataFrame, Dict]) -> Any:
        """
        Preprocess input data for model inference.
        
        Args:
            data: Input data (numpy array, DataFrame, or dict)
            
        Returns:
            Preprocessed data ready for model input
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, dict):
            # Handle dict input (e.g., from API request)
            # TODO: Implement proper dict preprocessing
            pass
        
        # Ensure correct shape and dtype
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        
        # Convert to appropriate format for model type
        if self.model_type == 'pytorch':
            data = torch.FloatTensor(data).to(self.device)
        
        return data
    
    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict]) -> np.ndarray:
        """
        Run prediction on input data.
        
        Args:
            data: Input data
            
        Returns:
            Model predictions (probabilities or class labels)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        preprocessed = self.preprocess(data)
        
        # Run prediction based on model type
        if self.model_type == 'keras':
            predictions = self.model.predict(preprocessed)
        elif self.model_type == 'pytorch':
            with torch.no_grad():
                predictions = self.model(preprocessed)
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return predictions
    
    def predict_proba(self, data: Union[np.ndarray, pd.DataFrame, Dict]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            data: Input data
            
        Returns:
            Probability distribution over classes
        """
        predictions = self.predict(data)
        
        # Ensure predictions are probabilities
        if self.model_type == 'pytorch':
            # Apply softmax if needed
            if predictions.shape[-1] > 1:
                predictions = torch.softmax(torch.FloatTensor(predictions), dim=-1).numpy()
        
        return predictions
    
    def predict_binary(self, data: Union[np.ndarray, pd.DataFrame, Dict]) -> np.ndarray:
        """
        Get binary class predictions (0 or 1).
        
        Args:
            data: Input data
            
        Returns:
            Binary class predictions
        """
        probas = self.predict_proba(data)
        
        # Get class with highest probability
        if probas.ndim > 1 and probas.shape[-1] > 1:
            predictions = np.argmax(probas, axis=-1)
        else:
            # Binary threshold
            predictions = (probas > 0.5).astype(int)
        
        return predictions


def load_model(model_path: str, device: str = 'cpu') -> InferencePipeline:
    """
    Convenience function to load a model into an InferencePipeline.
    
    Args:
        model_path: Path to model file
        device: Device for PyTorch models
        
    Returns:
        InferencePipeline with loaded model
    """
    pipeline = InferencePipeline(model_path=model_path, device=device)
    return pipeline


def preprocess_data(
    data: Union[np.ndarray, pd.DataFrame, Dict],
    target_shape: Optional[tuple] = None
) -> np.ndarray:
    """
    Standalone preprocessing function.
    
    Args:
        data: Input data
        target_shape: Optional target shape for reshaping
        
    Returns:
        Preprocessed numpy array
    """
    # Convert to numpy
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, dict):
        # Assume dict has 'data' key
        if 'data' in data:
            data = np.array(data['data'])
        else:
            raise ValueError("Dict input must have 'data' key")
    
    # Ensure float32
    data = data.astype(np.float32)
    
    # Reshape if needed
    if target_shape is not None:
        data = data.reshape(target_shape)
    
    return data
