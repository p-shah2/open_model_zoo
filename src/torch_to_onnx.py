import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

weights_path = '/home/016712345/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Load the weights manually from a .pth file
model.load_state_dict(torch.load(weights_path))

# Put the model in evaluation mode
model.eval()

# Create dummy input in the form of a batch of images
dummy_input = torch.randn(1, 3, 800, 800)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "fasterrcnn_resnet50_fpn.onnx",
                  opset_version=11,
                  input_names=['input'],
                  output_names=['output'],
                  do_constant_folding=True)



