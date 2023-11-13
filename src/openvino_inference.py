import argparse
from openvino.runtime import Core
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import time
import cv2

BATCH_SIZE = 1
num_threads = 4

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run OpenVINO model inference with dynamic model directory and device selection.')
    parser.add_argument('--model_directory', type=str, default='/home/016712345/models', help='The directory of the OpenVINO model files.')
    parser.add_argument('--model_base_name', type=str, default='person-detection-retail-0013', help='The base name of the OpenVINO model files without extension.')
    parser.add_argument('--precision', type=str, default='FP32', choices=['FP16', 'FP16-INT8', 'FP32'], help='The precision of the OpenVINO model files.')
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU'], help='The device to run the model on (CPU or GPU).')
    return parser.parse_args()

def preprocess_image_opencv(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image
    
# Function to preprocess image
def preprocess_image(image_path, w, h):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((w, h))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0).numpy()
    return image


def load_model(args):
    ie = Core()
    model_xml = f"{args.model_directory}/{args.model_base_name}/{args.precision}/{args.model_base_name}.xml"
    model_bin = f"{args.model_directory}/{args.model_base_name}/{args.precision}/{args.model_base_name}.bin"
    model = ie.read_model(model=model_xml, weights=model_bin)
    compiled_model = ie.compile_model(model=model, device_name=args.device)
    return compiled_model


def main():
    args = parse_arguments()
    compiled_model = load_model(args)
    # Create a list to hold the names of input and output layers
    input_layer = next(iter(compiled_model.inputs))
    output_layer = next(iter(compiled_model.outputs))

    n, c, h, w = input_layer.shape

    # print(f"Expected input width: {w}")
    # print(f"Expected input height: {h}")

    # Prepare the input batch
    input_image_path = "/home/016712345/models/images/n02099601_3004.jpg"  
    input_batch = preprocess_image(input_image_path, w, h)
    input_batch = np.repeat(input_batch, BATCH_SIZE, axis=0)

    # Perform inference
    print(f"Inference optimization on model: {args.model_base_name}")
    start_time = time.time()
    predictions = compiled_model([input_batch])[output_layer]
    end_time = time.time()
    duration = end_time - start_time
    print(f"OpenVINO {args.device} execution time: {duration} seconds")
    # Display the predictions
    # print(predictions)


if __name__ == "__main__":
    main()