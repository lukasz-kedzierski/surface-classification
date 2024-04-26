import torch

from models import CNNSurfaceClassifier

# Load the model for real-time inference
loaded_model = CNNSurfaceClassifier(input_size=None, output_size=None)
loaded_model.load_state_dict(torch.load('surface_classification_model.pth'))
loaded_model.eval()

prediction = loaded_model(new_data)
