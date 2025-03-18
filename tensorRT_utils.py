import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
from PIL import Image


class TensorRTInference:
    def __init__(self, onnx_file_path, engine_file_path=None):
        """
        Initialize TensorRT engine from ONNX model
        
        Args:
            onnx_file_path: Path to the ONNX model
            engine_file_path: Path to save/load TensorRT engine (optional)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.engine_file_path = engine_file_path or os.path.splitext(onnx_file_path)[0] + '.engine'
        
        # Load or create TensorRT engine
        if os.path.exists(self.engine_file_path):
            print(f"Loading existing TensorRT engine from {self.engine_file_path}")
            self.load_engine()
        else:
            print(f"Creating TensorRT engine from ONNX model {onnx_file_path}")
            self.build_engine_from_onnx(onnx_file_path)
            
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for input/output
        self.setup_io_binding()
    
    def build_engine_from_onnx(self, onnx_file_path):
        """Build TensorRT engine from ONNX model"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError(f"Failed to parse ONNX file {onnx_file_path}")
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Handle API changes in newer TensorRT versions
        # Set memory pool limit for workspace
        try:
            # 1GB workspace
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        except AttributeError:
            try:
                config.max_workspace_size = 1 << 30
            except AttributeError:
                print("Warning: Could not set workspace size. Please check your TensorRT version.")
        
        # Enable FP16 precision if available
        try:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        except AttributeError:
            try:
                config.set_flag(trt.BuilderFlag.FP16)
            except:
                print("Warning: Could not enable FP16. Please check your TensorRT version.")
        
        # Get input dimensions
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        print(f"Network input name: {input_name}, shape: {input_shape}")
        
        # Handle dynamic shapes if needed
        if any(dim == -1 for dim in input_shape):
            profile = builder.create_optimization_profile()
            
            # Example for dynamic batching
            if input_shape[0] == -1:  # Dynamic batch size
                min_batch = 1
                opt_batch = 1
                max_batch = 16
                min_shape = (min_batch,) + tuple(1 if d == -1 else d for d in input_shape[1:])
                opt_shape = (opt_batch,) + tuple(1 if d == -1 else d for d in input_shape[1:])
                max_shape = (max_batch,) + tuple(1 if d == -1 else d for d in input_shape[1:])
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
        
        # Build and save engine
        try:
            serialized_engine = builder.build_serialized_network(network, config)
            with open(self.engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        except AttributeError:
            # Fallback for older TensorRT versions
            self.engine = builder.build_engine(network, config)
            with open(self.engine_file_path, 'wb') as f:
                f.write(self.engine.serialize())
    
    def load_engine(self):
        """Load TensorRT engine from file"""
        with open(self.engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # Print engine information
        print(f"Loaded TensorRT engine: {self.engine_file_path}")
    
    def setup_io_binding(self):
        """Setup input and output bindings"""
        # Get input and output binding information
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        # For TensorRT 8.x+, we need to use different methods to get binding info
        try:
            # Get input details
            self.input_name = self.engine.get_tensor_name(0)
            self.input_shape = self.context.get_tensor_shape(self.input_name)
            
            # Get output details
            self.output_name = self.engine.get_tensor_name(1)
            self.output_shape = self.context.get_tensor_shape(self.output_name)
            
            print(f"TensorRT 8.x+ API: Input name: {self.input_name}, shape: {self.input_shape}")
            print(f"TensorRT 8.x+ API: Output name: {self.output_name}, shape: {self.output_shape}")
            
            # Create GPU buffers and stream
            self.stream = cuda.Stream()
           # Replace dynamic dimensions with concrete values for memory allocation
            concrete_input_shape = tuple(1 if dim == -1 else dim for dim in self.input_shape)
            concrete_output_shape = tuple(1 if dim == -1 else dim for dim in self.output_shape)

            # Allocate GPU memory for input and output
            self.d_input = cuda.mem_alloc(trt.volume(concrete_input_shape) * np.dtype(np.float32).itemsize)
            self.d_output = cuda.mem_alloc(trt.volume(concrete_output_shape) * np.dtype(np.float32).itemsize)
            
            # Create host buffers
            self.h_output = cuda.pagelocked_empty(trt.volume(concrete_output_shape), dtype=np.float32)
            
        except (AttributeError, RuntimeError) as e:
            print(f"Error using TensorRT 8.x+ API: {e}")
            print("Falling back to older TensorRT API...")
            
            # For older TensorRT versions, use num_bindings method
            num_bindings = self.engine.num_bindings
            for binding_idx in range(num_bindings):
                name = self.engine.get_binding_name(binding_idx)
                shape = self.engine.get_binding_shape(binding_idx)
                if self.engine.binding_is_input(binding_idx):
                    self.input_name = name
                    self.input_shape = shape
                    self.input_idx = binding_idx
                else:
                    self.output_name = name
                    self.output_shape = shape
                    self.output_idx = binding_idx
            
            print(f"Older TensorRT API: Input name: {self.input_name}, shape: {self.input_shape}")
            print(f"Older TensorRT API: Output name: {self.output_name}, shape: {self.output_shape}")
            
            # Create GPU buffers and stream
            self.stream = cuda.Stream()
            self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
            self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)
            self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
            self.bindings = [int(self.d_input), int(self.d_output)]
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to the input image
            target_size: Target size for resizing (H, W)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Determine target size based on input shape if not provided
        if target_size is None:
            if len(self.input_shape) == 4:  # NCHW format
                target_size = (self.input_shape[2], self.input_shape[3])
            else:
                target_size = (224, 224)  # Default size
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1] range
        img_array = img_array / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Rearrange to NCHW format (batch, channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
   #comment 
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: Input data as numpy array (NCHW format)
            
        Returns:
            Output data as numpy array
        """
        # Copy input data to GPU
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
        # Execute inference
        try:
            # TensorRT 8.x+ API
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            self.context.execute_async_v3(self.stream.handle)
        except (AttributeError, RuntimeError):
            # Older TensorRT API
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output back to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape output
        output = self.h_output.reshape(self.output_shape)
        
        return output
    
    def classify_image(self, image_path, class_names=None):
        """
        Classify an image using the model
        
        Args:
            image_path: Path to the input image
            class_names: List of class names (optional)
            
        Returns:
            Predicted class and confidence score
        """
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        
        # Run inference
        output = self.infer(input_data)
        
        # Get predicted class
        predictions = output.flatten()
        
        # Check if this is a classification output
        if len(predictions) > 1:
            # Classification
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            if class_names and predicted_class < len(class_names):
                return class_names[predicted_class], confidence
            else:
                return predicted_class, confidence
        else:
            # Not classification - return raw output
            return output
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'stream'):
            self.stream.synchronize()
            del self.stream
        if hasattr(self, 'd_input'):
            del self.d_input
        if hasattr(self, 'd_output'):
            del self.d_output


# Example usage
def main():
    # Path to your ONNX model
    onnx_model_path = "path/to/vision_model.onnx"
    
    # Optional: Path to ImageNet class names
    class_names_path = "path/to/imagenet_classes.txt"
    class_names = None
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    
    # Create inference engine with verbose output
    print("Creating TensorRT inference engine...")
    engine = TensorRTInference(onnx_model_path)
    
    # Path to test image
    test_image_path = "path/to/test_image.jpg"
    
    # Run inference
    print("Running inference on test image...")
    result = engine.classify_image(test_image_path, class_names)
    
    if isinstance(result, tuple):
        prediction, confidence = result
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print(f"Output: {result}")


if __name__ == "__main__":
    main()