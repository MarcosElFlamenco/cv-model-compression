import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from PIL import Image
from sklearn.cluster import DBSCAN
import json
import argparse
import os
from typing import Dict, Tuple, List, Optional, Union



#!/usr/bin/env python
"""
ImageNet Class Translation Script

This script provides utilities to translate numerical ImageNet class indices
to human-readable class names and descriptions.
"""


labels_file = "imagenet_labels.txt"

def print_top_n_results(topn_values,topn_indices):
    print("Predictions:")
    for i in range(len(topn_indices)):
        prediction = interpret_imagenet_index(labels_file,topn_indices[i])
        confidence = topn_values[i]
        print(f"{prediction} | {confidence}")
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
    return labels[idx]
            


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

