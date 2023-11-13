import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import time

BATCH_SIZE = 64
num_threads = 4

# Create an ONNX Runtime session options object
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = num_threads
sess_options.inter_op_num_threads = num_threads
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session = ort.InferenceSession("resnext50_onnx_model.onnx", sess_options, providers=['CUDAExecutionProvider'])
print("Available ONNX Runtime providers:", ort.get_available_providers())


# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Add batch dimension
    image = image.unsqueeze(0).numpy()
    return image

# Prepare the input batch
input_image_path = "/home/016712345/models/images/n02099601_3004.jpg"
input_batch = preprocess_image(input_image_path)

# Repeat the batch to match the BATCH_SIZE
BATCH_SIZE = 64
input_batch = np.repeat(input_batch, BATCH_SIZE, axis=0)

# Get the name of the input and output nodes
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Perform inference on the GPU
start_time = time.time()

# Run the model (forward pass)
predictions = ort_session.run([output_name], {input_name: input_batch})[0]
end_time = time.time()
duration = end_time - start_time
print(f"ONNX Runtime GPU execution time: {duration} seconds")

# Display the predictions
print(predictions)


