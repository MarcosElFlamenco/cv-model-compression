import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from PIL import Image
from sklearn.cluster import DBSCAN


#!/usr/bin/env python
"""
ImageNet Class Translation Script

This script provides utilities to translate numerical ImageNet class indices
to human-readable class names and descriptions.
"""

import json
import argparse
import os
from typing import Dict, Tuple, List, Optional, Union
import numpy as np


def interpret_imagenet_index(file_path,idx):
    labels = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
                
            # Split by colon and extract index and label
            parts = line.split(":", 1)
            index_str = parts[0].strip().strip("'")
            label = parts[1].strip().strip(",").strip("'")
            
            try:
                index = int(index_str)
                labels[index] = label
            except ValueError:
                continue
    
  #  # Convert dictionary to array
    #max_index = max(labels.keys())
    #labels_array = [""] * (max_index + 1)
    
    #for idx, label in labels.items():
        #labels_array[idx] = label
        
    #return labels_array
    #translation_file = "imagenet_labels.json"
    #with open(translation_file, 'r') as f:
        #labels = json.load(f):w
    print(labels[0])
            

class ImageNetTranslator: 
    """
    A utility class for translating ImageNet class indices to human-readable labels.
    """

    def __init__(self, labels_path: Optional[str] = None):
        """
        Initialize the ImageNet translator.
        
        Args:
            labels_path: Path to the ImageNet labels JSON file. If None, will use
                         built-in mappings.
        """
        self.class_idx_to_label = {}
        self.imagenet_labels_loaded = False
        
        # Load labels either from provided path or use built-in mappings
        if labels_path and os.path.exists(labels_path):
            self.load_labels_from_file(labels_path)
        else:
            self.load_built_in_labels()
    
    def load_built_in_labels(self):
        """Load the built-in ImageNet class mappings."""
        # A simplified subset of ImageNet labels for demonstration
        # In a real implementation, this would contain all 1000 classes
        self.class_idx_to_label = {
            0: ("n01440764", "tench", "Tinca tinca"),
            1: ("n01443537", "goldfish", "Carassius auratus"),
            2: ("n01484850", "great white shark", "white shark, Carcharodon carcharias"),
            3: ("n01491361", "tiger shark", "Galeocerdo cuvieri"),
            4: ("n01494475", "hammerhead shark", "hammerhead, Sphyrna zygaena"),
            5: ("n01496331", "electric ray", "electric ray, crampfish, numbfish, torpedo"),
            # ... more classes would be defined here
            996: ("n13054560", "bolete", ""),
            997: ("n13133613", "ear", "ear fungus"),
            998: ("n15075141", "toilet tissue", "toilet paper, bathroom tissue"),
            999: ("n13040303", "wooden spoon", ""),
        }
        self.imagenet_labels_loaded = True
        print("Loaded built-in ImageNet labels")
        
    def load_labels_from_file(self, filepath: str):
        """
        Load ImageNet class labels from a JSON file.
        
        Expected format:
        {
            "0": ["n01440764", "tench", "Tinca tinca"],
            ...
        }
        
        Args:
            filepath: Path to the JSON file with ImageNet labels
        """
        try:
            with open(filepath, 'r') as f:
                labels_dict = json.load(f)
                
            self.class_idx_to_label = {
                int(idx): tuple(values) for idx, values in labels_dict.items()
            }
            self.imagenet_labels_loaded = True
            print(f"Successfully loaded ImageNet labels from {filepath}")
        except Exception as e:
            print(f"Error loading labels from {filepath}: {e}")
            print("Falling back to built-in labels...")
            self.load_built_in_labels()
    
    def translate(self, class_idx: Union[int, np.ndarray, List[int]], 
                  return_wnid: bool = False, 
                  return_description: bool = False) -> Union[str, List[str], Tuple[str, ...]]:
        """
        Translate ImageNet class indices to human-readable labels.
        
        Args:
            class_idx: Integer class index (0-999) or list/array of indices
            return_wnid: Whether to include WordNet ID in the output
            return_description: Whether to include full description in the output
            
        Returns:
            Human-readable class name(s) or tuple with additional info if requested
        """
        if not self.imagenet_labels_loaded:
            raise ValueError("ImageNet labels have not been loaded")
        
        # Handle array-like inputs
        if isinstance(class_idx, (list, np.ndarray)):
            return [self._translate_single(idx, return_wnid, return_description) 
                   for idx in class_idx]
        else:
            return self._translate_single(class_idx, return_wnid, return_description)
    
    def _translate_single(self, class_idx: int, return_wnid: bool, return_description: bool):
        """Helper method to translate a single class index."""
        if class_idx not in self.class_idx_to_label:
            raise ValueError(f"Unknown class index: {class_idx}")
        
        wnid, label, description = self.class_idx_to_label[class_idx]
        
        if return_wnid and return_description:
            return (label, wnid, description)
        elif return_wnid:
            return (label, wnid)
        elif return_description:
            return (label, description)
        else:
            return label
    
    def translate_top_k(self, probabilities: np.ndarray, k: int = 5, 
                        return_wnid: bool = False, 
                        return_description: bool = False,
                        return_probability: bool = True) -> List:
        """
        Translate the top K class predictions from a model output.
        
        Args:
            probabilities: Array of class probabilities (shape [1000] or [N, 1000])
            k: Number of top classes to return
            return_wnid: Whether to include WordNet ID in results
            return_description: Whether to include full descriptions
            return_probability: Whether to include probability values
            
        Returns:
            List of top k predictions with requested information
        """
        if not self.imagenet_labels_loaded:
            raise ValueError("ImageNet labels have not been loaded")
            
        # Handle batch predictions
        if len(probabilities.shape) > 1:
            return [self._translate_top_k_single(probs, k, return_wnid, 
                                               return_description, return_probability) 
                   for probs in probabilities]
        else:
            return self._translate_top_k_single(probabilities, k, return_wnid, 
                                              return_description, return_probability)
    
    def _translate_top_k_single(self, probabilities, k, return_wnid, 
                              return_description, return_probability):
        """Helper method to translate top k predictions for a single sample."""
        top_k_idx = np.argsort(probabilities)[-k:][::-1]
        results = []
        
        for idx in top_k_idx:
            wnid, label, description = self.class_idx_to_label[idx]
            result = [label]
            
            if return_wnid:
                result.append(wnid)
            if return_description:
                result.append(description)
            if return_probability:
                result.append(float(probabilities[idx]))
                
            results.append(tuple(result) if len(result) > 1 else result[0])
            
        return results


def download_imagenet_labels(output_path: str = "imagenet_labels.json"):
    """
    Download the complete ImageNet class labels from a GitHub repository.
    
    Args:
        output_path: Path to save the labels JSON file
    """
    import requests
    
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert simple labels to the expected format
        simple_labels = response.json()
        full_labels = {}
        
        for i, label in enumerate(simple_labels):
            # Use placeholder WordNet IDs since we don't have them
            # In a real implementation, you would use actual WordNet IDs
            full_labels[str(i)] = [f"n{i:08d}", label, ""]
            
        with open(output_path, 'w') as f:
            json.dump(full_labels, f)
            
        print(f"Downloaded ImageNet labels to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading ImageNet labels: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Translate ImageNet class indices to human-readable labels")
    parser.add_argument("--class_idx", type=int, nargs="+", help="Class index or indices to translate")
    parser.add_argument("--probability_file", type=str, help="File containing model output probabilities (numpy format)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top classes to return")
    parser.add_argument("--labels_file", type=str, help="Path to ImageNet labels JSON file")
    parser.add_argument("--download_labels", action="store_true", help="Download complete ImageNet labels")
    parser.add_argument("--return_wnid", action="store_true", help="Include WordNet IDs in output")
    parser.add_argument("--return_description", action="store_true", help="Include full descriptions in output")
    
    args = parser.parse_args()
    
    if args.download_labels:
        output_path = args.labels_file or "imagenet_labels.json"
        download_imagenet_labels(output_path)
        return
    
    translator = ImageNetTranslator(args.labels_file)
    
    if args.probability_file:
        try:
            probs = np.load(args.probability_file)
            results = translator.translate_top_k(
                probs, 
                k=args.top_k,
                return_wnid=args.return_wnid,
                return_description=args.return_description
            )
            print(f"Top {args.top_k} predictions:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
        except Exception as e:
            print(f"Error processing probability file: {e}")
    elif args.class_idx:
        for idx in args.class_idx:
            result = translator.translate(
                idx,
                return_wnid=args.return_wnid,
                return_description=args.return_description
            )
            print(f"Class {idx}: {result}")
    else:
        print("Please provide either class indices or a probability file.")
        parser.print_help()



def apply_dbscan_to_vision_output(cov, bbox, eps=0.5, min_samples=5):
    """
    Apply DBSCAN clustering to the output tensors from a vision model.
    
    Parameters:
    -----------
    feature_tensor1 : numpy.ndarray
        First output tensor from the vision model (16x16xD)
    feature_tensor2 : numpy.ndarray
        Second output tensor from the vision model (16x16xE)
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
        
    Returns:
    --------
    labels : numpy.ndarray
        Cluster labels for each point in the dataset.
    clustered_data : numpy.ndarray
        Original data points with their cluster labels.
    """
    # Get the dimensions of the tensors
    num_classes, height, width = cov.shape
    dim2, _, _= bbox.shape
    
    # Reshape the tensors to 2D arrays where each row represents a pixel
    # Each row will contain features from both tensors

    cov_2d = cov.reshape(-1, num_classes)
    bbox_2d = bbox.reshape(-1, dim2)
    
    # Concatenate the features from both tensors
    combined_features = np.hstack((cov_2d, bbox_2d))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(combined_features)
    
    # Reshape labels back to the original image dimensions
    labels_2d = labels.reshape(height, width)
    
    # Create a result array with original data and cluster labels
    clustered_data = np.column_stack((combined_features, labels))
    
    return labels, labels_2d, clustered_data

def visualize_clusters(labels_2d, figsize=(10, 10)):
    """
    Visualize the clusters in a 2D grid.
    
    Parameters:
    -----------
    labels_2d : numpy.ndarray
        2D array of cluster labels.
    figsize : tuple, default=(10, 10)
        Figure size.
    """
    plt.figure(figsize=figsize)
    plt.imshow(labels_2d, cmap='viridis')
    plt.colorbar(label='Cluster Label')
    plt.title('DBSCAN Clustering Results')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Assuming your vision model outputs two tensors of shape (16, 16, D) and (16, 16, E)
    # where D and E are the feature dimensions
    # This is just an example with random data
    D = 64  # Example feature dimension for first tensor
    E = 32  # Example feature dimension for second tensor
    
    # Generate random example data
    feature_tensor1 = np.random.rand(16, 16, D)
    feature_tensor2 = np.random.rand(16, 16, E)
    
    # Apply DBSCAN
    labels, labels_2d, clustered_data = apply_dbscan_to_vision_output(
        feature_tensor1, feature_tensor2, 
        eps=0.5,  # Adjust this parameter based on your data
        min_samples=5  # Adjust this parameter based on your data
    )
    
    # Print some information about the clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {list(labels).count(-1)}")
    
    # Visualize the clusters
    visualize_clusters(labels_2d)

def process_model_outputs(class_convictions, box_vectors, input_image_shape, confidence_threshold=0.5):
    """
    Process model outputs to get bounding boxes
    
    Args:
        class_convictions: Array of shape (34, 60, 4) containing class probabilities
        box_vectors: Array of shape (34, 60, 16) containing bounding box information
        input_image_shape: Tuple (height, width) of the original input image
        confidence_threshold: Minimum confidence to consider a detection
        
    Returns:
        List of dictionaries with detected objects (class_id, confidence, bbox)
    """
    class_convictions = np.transpose(class_convictions ,(1,2,0))
    box_vectors = np.transpose(box_vectors ,(1,2,0))

    image_height, image_width = input_image_shape
    grid_height, grid_width = class_convictions.shape[:2]
    
    # Cell size in the original image
    cell_height = image_height / grid_height
    cell_width = image_width / grid_width
    
    # Scale factor from objective_set.bbox.scale in the config
    scale_factor = 35.0
    
    # Offset from objective_set.bbox.offset in the config
    offset = 0.5
    
    detections = []
    
    # Iterate through each cell in the grid
    for y in range(grid_height):
        for x in range(grid_width):
            # Get class confidences for this cell
            class_confidences = class_convictions[y, x]
            
            # Find the class with highest confidence
            class_id = np.argmax(class_confidences)
            confidence = class_confidences[class_id]
            
            # Skip if confidence is below threshold
            if confidence < confidence_threshold:
                continue
            
            # Get box vector for this cell
            # The box vector is interpreted as 4 sets of (x1, y1, x2, y2) coordinates, one for each class
            # So we need to extract the coordinates for the detected class
            box_data = box_vectors[y, x]
            
            # Each class has 4 values (x1, y1, x2, y2), so get the values for the detected class
            class_box_indices = slice(class_id * 4, (class_id + 1) * 4)
            x1, y1, x2, y2 = box_data[class_box_indices]
            
            # Apply scale and offset as per the model configuration
            # These transformations convert the normalized box coordinates to pixel coordinates
            # relative to the cell position
            x1 = (x1 / scale_factor) + offset
            y1 = (y1 / scale_factor) + offset
            x2 = (x2 / scale_factor) + offset
            y2 = (y2 / scale_factor) + offset
            
            # Convert to absolute coordinates in the original image
            abs_x1 = (x + x1) * cell_width
            abs_y1 = (y + y1) * cell_height
            abs_x2 = (x + x2) * cell_width
            abs_y2 = (y + y2) * cell_height
            
            # Ensure coordinates are within image boundaries
            abs_x1 = max(0, min(abs_x1, image_width - 1))
            abs_y1 = max(0, min(abs_y1, image_height - 1))
            abs_x2 = max(0, min(abs_x2, image_width - 1))
            abs_y2 = max(0, min(abs_y2, image_height - 1))
            
            # Store the detection
            detections.append({
                'class_id': class_id,
                'confidence': float(confidence),
                'bbox': [int(abs_x1), int(abs_y1), int(abs_x2 - abs_x1), int(abs_y2 - abs_y1)]  # [x, y, width, height]
            })
    
    return detections

def visualize_detections(image, detections, class_names=None):
    """
    Visualize detections on the image
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names (if None, will use class indices)
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(4)]
    
    # Colors for different classes (BGR format for OpenCV)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    
    # Create a copy of the image to draw on
    vis_image = image.copy()
    
    for det in detections:
        class_id = det['class_id']
        confidence = det['confidence']
        x, y, w, h = det['bbox']
        
        # Draw rectangle
        color = colors[class_id % len(colors)]
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert from BGR to RGB for matplotlib
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_image_rgb)
    plt.axis('off')
    plt.show()



def visualize_predictions_grid(image_path, output_tensor, confidence_threshold=0.5):
    """
    Visualizes the predictions of a grid-based vision model, only showing numbers for confident predictions.

    Args:
        image_path (str): Path to the input image.
        output_tensor (np.ndarray): Model output of shape (num_classes, 34, 60).
        confidence_threshold (float): Minimum confidence for displaying predictions.
    """
    # Constants
    GRID_ROWS, GRID_COLS = 34, 60
    CELL_SIZE = 16  # Each cell corresponds to 16x16 pixels
    num_classes, height, width = output_tensor.shape
    
    # Load and resize the image to match the grid structure
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.resize(image, (GRID_COLS * CELL_SIZE, GRID_ROWS * CELL_SIZE))  # Resize to match grid

    # Convert to RGB (OpenCV loads in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the best class and confidence score per cell
    class_predictions = np.argmax(output_tensor, axis=0)  # Shape: (34, 60)
    confidence_scores = np.max(output_tensor, axis=0)  # Get max confidence per cell

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    # Overlay grid and predictions
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Compute the position of the current grid cell
            x, y = col * CELL_SIZE, row * CELL_SIZE
            conf = confidence_scores[row, col]
            
            # Draw grid lines
            ax.add_patch(plt.Rectangle((x, y), CELL_SIZE, CELL_SIZE, edgecolor='red', linewidth=0.5, fill=False))

            # Only display class prediction if confidence is above the threshold
            if conf > confidence_threshold:
                pred_class = int(class_predictions[row, col])  # Get class prediction
                ax.text(x + CELL_SIZE / 2, y + CELL_SIZE / 2, str(pred_class),
                        color='white', fontsize=6, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    #ax.text(a + CELL_SIZE / 2, a + CELL_SIZE / 2, 'A',color='white', fontsize=16, ha='center', va='center',        bbox=dict(facecolor='black', alpha=1, edgecolor='none', boxstyle='round,pad=0.1'))
 

    ax.axis("off")  # Hide axes
    plt.title("Grid-Based Predictions (Thresholded)")
    plt.show()


def image_to_tensor(image_path,width=960,height=544):
    """
    Convert an image into a NumPy tensor with shape (1, 3, 544, 960).
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Image tensor of shape (1, 3, 544, 960).
    """
    # Load image with OpenCV (BGR format)
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize image to (960, 544) (Width x Height)
    image_resized = cv2.resize(image, (width, height))
     

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1] (optional, depending on your needs)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    # Change shape from (544, 960, 3) to (3, 544, 960)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    # Add batch dimension to get (1, 3, 544, 960)
    image_tensor = np.expand_dims(image_transposed, axis=0)

    return image_tensor


def show_image_from_tensor(tensor):
    """
    Displays an image from a NumPy tensor of shape (1, 3, 544, 960).
    
    Args:
        tensor (np.ndarray): Input image tensor of shape (1, 3, 544, 960).
    """
#    if tensor.shape != (1, 3, 544, 960):
#        raise ValueError(f"Expected tensor shape (1, 3, 544, 960), but got {tensor.shape}")

    # Remove batch dimension (1, 3, 544, 960) -> (3, 544, 960)
    image = tensor.squeeze(0)

    # Change from (3, H, W) to (H, W, 3) for displaying
    image = np.transpose(image, (1, 2, 0))

    # Clip values in case of out-of-range values (optional, for safety)
    image = np.clip(image, 0, 1)  # Assuming it's normalized to [0,1]

    # Display image
    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()

