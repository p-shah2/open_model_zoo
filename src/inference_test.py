import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image

# Step 1: Load a pretrained object detection model (Faster R-CNN)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Step 2: Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using ", device)
# Step 3: Optimize the model for inference
model.eval()

# Function to transform input image
def transform_image(image):
    image = F.to_tensor(image).unsqueeze(0)  # Transform the image to tensor and add batch dimension
    return image.to(device)

# Step 4: Prepare an input image 
image = Image.open("/home/016712345/yolov5/test_image.jpg")
input_image = transform_image(image)

# Step 5: Perform inference
with torch.no_grad():
    prediction = model(input_image)

print(prediction)


