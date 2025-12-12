"""
Model Explainability System for Chest X-Ray Pneumonia Detection
Provides GRAD-CAM visualizations and model interpretation capabilities
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import base64
import io
import json
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    transforms = None
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    cm = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Result of model explanation"""
    prediction_id: str
    model_id: str
    model_version: str
    explanation_type: str
    confidence_score: float
    predicted_class: str
    explanation_data: Dict[str, Any]
    visualization_data: Optional[str] = None  # Base64 encoded image
    metadata: Dict[str, Any] = None
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "explanation_type": self.explanation_type,
            "confidence_score": self.confidence_score,
            "predicted_class": self.predicted_class,
            "explanation_data": self.explanation_data,
            "visualization_data": self.visualization_data,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat()
        }

class GradCAM:
    """Gradient-weighted Class Activation Mapping implementation"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GRAD-CAM")
        
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
        logger.info(f"GradCAM initialized for layer: {target_layer}")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
        else:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor, 
                    target_class: int = None) -> np.ndarray:
        """Generate Class Activation Map"""
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class ModelExplainer:
    """Main model explainability system"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model explanation")
        
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ["NORMAL", "PNEUMONIA"]
        
        logger.info("ModelExplainer initialized")
    
    def explain_prediction(self, 
                          image: Union[np.ndarray, str, bytes],
                          prediction_id: str,
                          model_id: str,
                          model_version: str,
                          target_layer: str = None,
                          explanation_types: List[str] = None) -> ExplanationResult:
        """Generate comprehensive explanation for a prediction"""
        
        if explanation_types is None:
            explanation_types = ["gradcam", "attention", "feature_importance"]
        
        # Preprocess image
        input_tensor, original_image = self._preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class_idx = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
            predicted_class = self.class_names[predicted_class_idx]
        
        explanation_data = {}
        visualization_data = None
        
        # Generate GRAD-CAM if requested
        if "gradcam" in explanation_types:
            try:
                gradcam_result = self._generate_gradcam(
                    input_tensor, original_image, predicted_class_idx, target_layer
                )
                explanation_data["gradcam"] = gradcam_result["data"]
                visualization_data = gradcam_result["visualization"]
            except Exception as e:
                logger.error(f"Failed to generate GRAD-CAM: {e}")
                explanation_data["gradcam"] = {"error": str(e)}
        
        # Generate attention maps if requested
        if "attention" in explanation_types:
            try:
                attention_result = self._generate_attention_map(input_tensor, original_image)
                explanation_data["attention"] = attention_result
            except Exception as e:
                logger.error(f"Failed to generate attention map: {e}")
                explanation_data["attention"] = {"error": str(e)}
        
        # Generate feature importance if requested
        if "feature_importance" in explanation_types:
            try:
                feature_importance = self._generate_feature_importance(input_tensor)
                explanation_data["feature_importance"] = feature_importance
            except Exception as e:
                logger.error(f"Failed to generate feature importance: {e}")
                explanation_data["feature_importance"] = {"error": str(e)}
        
        # Create explanation result
        result = ExplanationResult(
            prediction_id=prediction_id,
            model_id=model_id,
            model_version=model_version,
            explanation_type="comprehensive",
            confidence_score=confidence,
            predicted_class=predicted_class,
            explanation_data=explanation_data,
            visualization_data=visualization_data,
            metadata={
                "input_shape": list(input_tensor.shape),
                "model_output_shape": list(output.shape),
                "explanation_types": explanation_types,
                "target_layer": target_layer
            }
        )
        
        return result
    
    def _preprocess_image(self, image: Union[np.ndarray, str, bytes]) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess image for model input"""
        
        # Handle different input types
        if isinstance(image, str):
            # File path
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, bytes):
            # Bytes data
            nparr = np.frombuffer(image, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # NumPy array
            original_image = image.copy()
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                # Assume BGR, convert to RGB
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to PIL Image for preprocessing
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(original_image)
            input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        else:
            # Manual preprocessing without PIL
            resized = cv2.resize(original_image, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # Convert to tensor
            input_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        return input_tensor, original_image
    
    def _generate_gradcam(self, 
                         input_tensor: torch.Tensor,
                         original_image: np.ndarray,
                         target_class: int,
                         target_layer: str = None) -> Dict[str, Any]:
        """Generate GRAD-CAM visualization"""
        
        # Determine target layer if not specified
        if target_layer is None:
            target_layer = self._find_best_target_layer()
        
        # Create GRAD-CAM instance
        gradcam = GradCAM(self.model, target_layer)
        
        try:
            # Generate CAM
            cam = gradcam.generate_cam(input_tensor, target_class)
            
            # Resize CAM to original image size
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            # Create heatmap visualization
            visualization = self._create_heatmap_visualization(original_image, cam_resized)
            
            # Calculate important regions
            important_regions = self._extract_important_regions(cam_resized)
            
            result = {
                "data": {
                    "target_layer": target_layer,
                    "target_class": target_class,
                    "cam_values": cam.tolist(),
                    "important_regions": important_regions,
                    "max_activation": float(cam.max()),
                    "min_activation": float(cam.min()),
                    "mean_activation": float(cam.mean())
                },
                "visualization": visualization
            }
            
            return result
            
        finally:
            gradcam.cleanup()
    
    def _generate_attention_map(self, 
                               input_tensor: torch.Tensor,
                               original_image: np.ndarray) -> Dict[str, Any]:
        """Generate attention-based explanation"""
        
        # This is a simplified attention mechanism
        # In practice, you might use integrated gradients or other methods
        
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        
        # Compute gradients
        self.model.zero_grad()
        output[0, predicted_class].backward()
        
        # Get input gradients
        gradients = input_tensor.grad.data
        
        # Calculate attention as absolute gradients
        attention = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Resize to original image size
        attention_resized = cv2.resize(attention, (original_image.shape[1], original_image.shape[0]))
        
        return {
            "attention_map": attention.tolist(),
            "max_attention": float(attention.max()),
            "min_attention": float(attention.min()),
            "mean_attention": float(attention.mean()),
            "attention_regions": self._extract_important_regions(attention_resized, threshold=0.7)
        }
    
    def _generate_feature_importance(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Generate feature importance analysis"""
        
        # Get model features at different layers
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks for key layers
        hooks = []
        layer_names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                layer_names.append(name)
                if len(hooks) >= 5:  # Limit to 5 layers
                    break
        
        try:
            # Forward pass
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Analyze features
            feature_analysis = {}
            
            for layer_name in layer_names:
                if layer_name in features:
                    feature_tensor = features[layer_name]
                    
                    # Calculate statistics
                    feature_analysis[layer_name] = {
                        "shape": list(feature_tensor.shape),
                        "mean_activation": float(feature_tensor.mean()),
                        "max_activation": float(feature_tensor.max()),
                        "min_activation": float(feature_tensor.min()),
                        "std_activation": float(feature_tensor.std()),
                        "sparsity": float((feature_tensor == 0).float().mean())
                    }
            
            return {
                "layer_analysis": feature_analysis,
                "total_layers_analyzed": len(feature_analysis),
                "model_complexity": self._calculate_model_complexity()
            }
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    def _find_best_target_layer(self) -> str:
        """Find the best layer for GRAD-CAM visualization"""
        
        # Look for the last convolutional layer
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        
        if conv_layers:
            return conv_layers[-1]  # Return last conv layer
        
        # Fallback to any layer with spatial dimensions
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) > 2:
                return name
        
        # Final fallback
        return "features"
    
    def _create_heatmap_visualization(self, 
                                    original_image: np.ndarray,
                                    cam: np.ndarray,
                                    alpha: float = 0.4) -> str:
        """Create heatmap overlay visualization"""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Heatmap
        heatmap = cm.jet(cam)[:, :, :3]  # Remove alpha channel
        axes[1].imshow(heatmap)
        axes[1].set_title("GRAD-CAM Heatmap")
        axes[1].axis('off')
        
        # Overlay
        overlay = original_image.astype(np.float32) / 255.0
        heatmap_overlay = alpha * heatmap + (1 - alpha) * overlay
        axes[2].imshow(heatmap_overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close(fig)
        
        return image_base64
    
    def _extract_important_regions(self, 
                                  activation_map: np.ndarray,
                                  threshold: float = 0.5,
                                  min_area: int = 100) -> List[Dict[str, Any]]:
        """Extract important regions from activation map"""
        
        # Threshold the activation map
        binary_map = (activation_map > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region statistics
                region_mask = np.zeros_like(activation_map)
                cv2.fillPoly(region_mask, [contour], 1)
                
                region_values = activation_map[region_mask == 1]
                
                regions.append({
                    "region_id": i,
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "area": int(area),
                    "mean_activation": float(region_values.mean()),
                    "max_activation": float(region_values.max()),
                    "centroid": {
                        "x": int(x + w // 2),
                        "y": int(y + h // 2)
                    }
                })
        
        # Sort by mean activation (most important first)
        regions.sort(key=lambda r: r["mean_activation"], reverse=True)
        
        return regions
    
    def _calculate_model_complexity(self) -> Dict[str, Any]:
        """Calculate model complexity metrics"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count layers by type
        layer_counts = {}
        for module in self.model.modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layer_counts": layer_counts,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class ExplainabilityService:
    """Service for managing model explanations"""
    
    def __init__(self, model_registry_path: str = "models"):
        self.model_registry_path = Path(model_registry_path)
        self.loaded_models = {}
        self.explainers = {}
        
        logger.info("ExplainabilityService initialized")
    
    def load_model(self, model_id: str, model_version: str) -> bool:
        """Load model for explanation"""
        
        model_key = f"{model_id}:{model_version}"
        
        if model_key in self.loaded_models:
            return True
        
        try:
            # Load model (this would integrate with your model registry)
            model_path = self.model_registry_path / model_id / model_version / "model.pth"
            
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False
            
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load(model_path, map_location=device)
            model.eval()
            
            # Create explainer
            explainer = ModelExplainer(model, device)
            
            self.loaded_models[model_key] = model
            self.explainers[model_key] = explainer
            
            logger.info(f"Loaded model for explanation: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return False
    
    def explain_prediction(self, 
                          image: Union[np.ndarray, str, bytes],
                          prediction_id: str,
                          model_id: str,
                          model_version: str,
                          explanation_types: List[str] = None) -> Optional[ExplanationResult]:
        """Generate explanation for a prediction"""
        
        model_key = f"{model_id}:{model_version}"
        
        # Load model if not already loaded
        if model_key not in self.explainers:
            if not self.load_model(model_id, model_version):
                return None
        
        try:
            explainer = self.explainers[model_key]
            result = explainer.explain_prediction(
                image=image,
                prediction_id=prediction_id,
                model_id=model_id,
                model_version=model_version,
                explanation_types=explanation_types
            )
            
            logger.info(f"Generated explanation for prediction {prediction_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return None
    
    def get_model_info(self, model_id: str, model_version: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        
        model_key = f"{model_id}:{model_version}"
        
        if model_key not in self.loaded_models:
            return {}
        
        model = self.loaded_models[model_key]
        explainer = self.explainers[model_key]
        
        return {
            "model_id": model_id,
            "model_version": model_version,
            "device": explainer.device,
            "class_names": explainer.class_names,
            "complexity": explainer._calculate_model_complexity(),
            "loaded_at": datetime.now().isoformat()
        }
    
    def cleanup_model(self, model_id: str, model_version: str):
        """Cleanup loaded model to free memory"""
        
        model_key = f"{model_id}:{model_version}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
        
        if model_key in self.explainers:
            del self.explainers[model_key]
        
        # Force garbage collection
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        
        logger.info(f"Cleaned up model: {model_key}")

# Global explainability service instance
_explainability_service = None

def get_explainability_service(model_registry_path: str = "models") -> ExplainabilityService:
    """Get global explainability service instance"""
    global _explainability_service
    
    if _explainability_service is None:
        _explainability_service = ExplainabilityService(model_registry_path)
    
    return _explainability_service