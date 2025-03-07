import cv2
import numpy as np
import matplotlib.pyplot as plt



def draw_bounding_boxes(image_path, bounding_boxes):
    """
    Draws bounding boxes on an image.

    Args:
        image_path (str): Path to the input image.
        bounding_boxes (list): List of bounding boxes in the format:
                               [(x_min, y_min, x_max, y_max, class_id, confidence), ...]
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert image to RGB (since OpenCV loads it in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    # Define colors for bounding boxes
    colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "white"]

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max, class_id, confidence = box

        # Choose a color based on class ID
        color = colors[class_id % len(colors)]

        # Draw the bounding box
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   edgecolor=color, linewidth=2, fill=False))

        # Add label with class and confidence
        label = f"Class {class_id} ({confidence:.2f})"
        ax.text(x_min, y_min - 5, label, color="white", fontsize=8, 
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.7))

    ax.axis("off")  # Hide axes
    plt.title("Detected Objects with Bounding Boxes")
    plt.show()



def extract_bounding_boxes(output_tensor, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Converts a grid-based model output into bounding boxes using thresholding and Non-Maximum Suppression (NMS).

    Args:
        output_tensor (np.ndarray): Model output of shape (num_classes, 34, 60).
        confidence_threshold (float): Minimum confidence to keep a prediction.
        nms_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        List of final bounding boxes [(x_min, y_min, x_max, y_max, class_id, confidence)].
    """
    GRID_ROWS, GRID_COLS = 34, 60  # Grid size
    CELL_SIZE = 16  # Each cell corresponds to a 16x16 region
    num_classes, height, width = output_tensor.shape
    
    # Step 1: Get the best class for each grid cell
    class_predictions = np.argmax(output_tensor, axis=0)  # Shape: (34, 60)
    confidence_scores = np.max(output_tensor, axis=0)  # Get max confidence per cell

    # Step 2: Filter cells with high confidence
    boxes = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            conf = confidence_scores[row, col]
            if conf > confidence_threshold:
                class_id = int(class_predictions[row, col])
                
                # Compute bounding box coordinates in image space
                x_min = col * CELL_SIZE
                y_min = row * CELL_SIZE
                x_max = x_min + CELL_SIZE
                y_max = y_min + CELL_SIZE

                boxes.append((x_min, y_min, x_max, y_max, class_id, conf))

    # Step 3: Apply Non-Maximum Suppression (NMS)
    if len(boxes) == 0:
        return []  # No detections

    # Convert boxes to OpenCV format: [x, y, width, height]
    boxes_np = np.array([[x_min, y_min, x_max - x_min, y_max - y_min, conf] for x_min, y_min, x_max, y_max, _, conf in boxes])
    class_ids = np.array([c for _, _, _, _, c, _ in boxes])

    indices = cv2.dnn.NMSBoxes(boxes_np[:, :4].tolist(), boxes_np[:, 4].tolist(), confidence_threshold, nms_threshold)
    print(indices)
    
    # Filter final bounding boxes
    final_boxes = [boxes[i] for i in indices] if len(indices) > 0 else []

    return final_boxes



def visualize_predictions(image_path, output_tensor, confidence_threshold=0.5):
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

    ax.axis("off")  # Hide axes
    plt.title("Grid-Based Predictions (Thresholded)")
    plt.show()


def image_to_tensor(image_path):
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
    image_resized = cv2.resize(image, (960, 544))

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
    if tensor.shape != (1, 3, 544, 960):
        raise ValueError(f"Expected tensor shape (1, 3, 544, 960), but got {tensor.shape}")

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

