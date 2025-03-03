import os
import numpy as np
import cv2
from tlt.models import detect, classify, segment  # Import appropriate module based on your model type
import argparse

class SimpleTLTInference:
    def __init__(self, tlt_model_path, model_type='detectnet_v2', key=None):
        """
        Initialize TLT inference
        
        Args:
            tlt_model_path: Path to the .tlt model file
            model_type: Type of model (detectnet_v2, classification, etc.)
            key: Encryption key used during training (if applicable)
        """
        self.model_path = tlt_model_path
        self.model_type = model_type
        self.key = key
        
        # Load the model based on type
        if 'detect' in model_type.lower():
            # For detection models like detectnet_v2, yolo, etc.
            self.model = detect.TLTDetect(tlt_model_path, key=key)
        elif 'classif' in model_type.lower():
            # For classification models
            self.model = classify.TLTClassify(tlt_model_path, key=key)
        elif 'segment' in model_type.lower():
            # For segmentation models
            self.model = segment.TLTSegment(tlt_model_path, key=key)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Get model input shape
        self.input_shape = self.model.input_shape
        
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image ready for model input
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
            
        # Resize to model's input dimensions
        h, w = self.input_shape[1:3]  # Height and width from input shape
        resized_image = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB (if needed)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess based on model type (normalization, etc.)
        # This is handled internally by the TLT API
        
        return image, rgb_image
    
    def run_inference(self, image_path):
        """
        Run inference on an image
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Original image and inference results
        """
        # Preprocess the image
        original_image, preprocessed_image = self.preprocess_image(image_path)
        
        # Run inference (the API handles all the preprocessing internally)
        if 'detect' in self.model_type.lower():
            # For detection models
            detections = self.model.infer(preprocessed_image)
            return original_image, detections
        
        elif 'classif' in self.model_type.lower():
            # For classification models
            classifications = self.model.infer(preprocessed_image)
            return original_image, classifications
            
        elif 'segment' in self.model_type.lower():
            # For segmentation models
            segmentation = self.model.infer(preprocessed_image)
            return original_image, segmentation
    
    def visualize_results(self, image, results):
        """
        Visualize inference results on the image
        
        Args:
            image: Original image
            results: Inference results
            
        Returns:
            Image with visualized results
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        if 'detect' in self.model_type.lower():
            # Visualize detection results
            for detection in results:
                class_id = detection['class_id']
                confidence = detection['confidence']
                bbox = detection['bbox']  # [xmin, ymin, xmax, ymax]
                
                # Convert bbox coordinates to integers
                xmin, ymin, xmax, ymax = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw label
                label = f"Class {class_id}: {confidence:.2f}"
                cv2.putText(vis_image, label, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        elif 'classif' in self.model_type.lower():
            # Visualize classification results
            top_k = 3  # Show top-k predictions
            for i, (class_id, confidence) in enumerate(results[:top_k]):
                label = f"Class {class_id}: {confidence:.2f}"
                cv2.putText(vis_image, label, (10, 30 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        elif 'segment' in self.model_type.lower():
            # Visualize segmentation results (overlay mask)
            mask = results['mask']
            # Resize mask to match original image size
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            for class_id in np.unique(mask_resized):
                if class_id == 0:  # Skip background
                    continue
                # Random color for each class
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask[mask_resized == class_id] = color
                
            # Overlay mask on image
            alpha = 0.5
            vis_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
            
        return vis_image

# Example usage
def infer(tlt_model_path,image_path):
    # TLT model encryption key (if applicable)
    model_key = "tlt_encode"

    # Initialize the inference
    tlt_inference = SimpleTLTInference(
        tlt_model_path=tlt_model_path,
        model_type="detectnet_v2",  # Change to your model type
        key=model_key
    )
    
    # Run inference
    original_image, results = tlt_inference.run_inference(image_path)
    
    # Visualize and save results
    visualization = tlt_inference.visualize_results(original_image, results)
    cv2.imwrite("inference_result.jpg", visualization)
    
    # Print results
    print("Inference Results:")
    print(results)