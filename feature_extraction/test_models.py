import torch
from models import load_all_models

# Load models
models_dict, device = load_all_models()

print("Devices:", device)

# Create a sample image tensor
sample_image = torch.randn(1, 3, 224,224).to(device)

# Test feature extraction
with torch.no_grad():

    for model_name, model in models_dict.items():

        features = model(sample_image)

        print(
            f"{model_name}:"
            f"Feature shape = {features.shape}"
        )

print("\nAll models tested successfully!")        