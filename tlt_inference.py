import os
import numpy as np
import cv2
import tensorrt as trt
from cuda import cudart
import pycuda.driver as cuda
import pycuda.autoinit

class TLTInference:
    def __init__(self, engine_path):
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(binding)
            binding_size = trt.volume(binding_shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(binding_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to bindings
            self.bindings.append(int(device_mem))
            
            # Append to inputs/outputs list
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": binding_shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": binding_shape})
    
    def preprocess(self, img_path, input_shape):
        """Preprocess the input image to match model input requirements"""
        # Read image
        img = cv2.imread(img_path)
        
        # Resize to expected input dimensions
        target_height, target_width = input_shape[1], input_shape[2]
        img_resized = cv2.resize(img, (target_width, target_height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] or [-1,1] depending on your model's requirements
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW format if required (common in deep learning)
        img_nchw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_nchw, axis=0)
        
        return img_batch
    
    def infer(self, input_data):
        """Run inference with the TLT model"""
        # Copy input data to input buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        
        # Transfer input data to GPU
        for inp in self.inputs:
            cuda.memcpy_htod(inp["device"], inp["host"])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Transfer outputs from GPU to CPU
        for out in self.outputs:
            cuda.memcpy_dtoh(out["host"], out["device"])
        
        # Return outputs
        return [out["host"].reshape(out["shape"]) for out in self.outputs]
    
    def postprocess(self, outputs, threshold=0.5):
        """Process model outputs based on the model type"""
        # This will vary based on your model type (detection, classification, etc.)
        # Example for detection model:
        detections = []
        
        if len(outputs) >= 1:  # Assuming outputs[0] contains detection results
            # Format: [batch_id, class_id, confidence, xmin, ymin, xmax, ymax]
            for detection in outputs[0]:
                confidence = detection[2]
                
                if confidence >= threshold:
                    class_id = int(detection[1])
                    bbox = detection[3:7]  # [xmin, ymin, xmax, ymax]
                    detections.append({"class_id": class_id, "confidence": confidence, "bbox": bbox})
        
        return detections

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate LLM against Stockfish to compute Elo.")
    parser.add_argument('--model_path', type=str,nargs="+", help='Path to model checkpoint.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory with meta.pkl.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to play.')
    parser.add_argument('--time_per_move', type=float, default=0.1, help='Time per move for Stockfish (seconds).')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for invalid LLM moves.')
    parser.add_argument('--evaluation_games', type=int, default=3, help='Max retries for invalid LLM moves.')
    # Path to your TLT exported engine
    engine_path = "path/to/your/model.engine"
    
    # Initialize the inference class
    tlt_model = TLTInference(engine_path)
    
    # Path to image for inference
    image_path = "path/to/your/image.jpg"
    
    # Get input shape from the model (assuming first binding is input)
    input_shape = tlt_model.inputs[0]["shape"]
    
    # Preprocess the image
    preprocessed_img = tlt_model.preprocess(image_path, input_shape)
    
    # Run inference
    outputs = tlt_model.infer(preprocessed_img)
    
    # Post-process the results
    detections = tlt_model.postprocess(outputs, threshold=0.5)
    
    # Print results
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"Detection {i+1}: Class ID {det['class_id']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")