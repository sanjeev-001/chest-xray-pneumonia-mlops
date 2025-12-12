#!/usr/bin/env python3
"""
Quick script to inspect the checkpoint and test loading
"""
import torch
from torchvision.models import efficientnet_b4

# Load the checkpoint
checkpoint = torch.load("models/best_chest_xray_model.pth", map_location="cpu")
print("Checkpoint keys (first 10):")
keys = list(checkpoint.keys())
for i, key in enumerate(keys[:10]):
    print(f"  {i+1}. {key}")

print(f"\nTotal keys: {len(keys)}")

# Test the EXACT architecture from the API
print("\nTrying API architecture...")
model = efficientnet_b4(weights=None)
import torch.nn as nn

# Use the EXACT same classifier as the API
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),  # 0
    nn.Linear(1792, 512),             # 1 - matches classifier.1 in checkpoint
    nn.ReLU(inplace=True),            # 2
    nn.BatchNorm1d(512),              # 3 - matches classifier.3 in checkpoint  
    nn.Dropout(p=0.2, inplace=True), # 4
    nn.Linear(512, 2)                 # 5 - matches classifier.5 in checkpoint
)

print(f"API classifier: {model.classifier}")

try:
    model.load_state_dict(checkpoint, strict=True)
    print("✅ SUCCESS: Checkpoint matches API architecture!")
    
    # Test with dummy input (EfficientNet-B4 optimal size)
    dummy_input = torch.randn(1, 3, 380, 380)
    with torch.no_grad():
        model.eval()
        output = model(dummy_input)
        print(f"✅ Model works! Output shape: {output.shape}")
        print(f"✅ Raw output: {output.numpy()}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")