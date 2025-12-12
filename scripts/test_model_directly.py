#!/usr/bin/env python3
"""
Direct model test to diagnose the class order issue
Tests the model with actual images to see what it outputs
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Create model exactly as in API
model = efficientnet_b4(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(1792, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(512, 2)
)

# Load checkpoint
checkpoint_path = "models/best_chest_xray_model.pth"
if not os.path.exists(checkpoint_path):
    print(f"❌ Model file not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Handle checkpoint format
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Fix key mismatch
if any(key.startswith('backbone.features.') for key in state_dict.keys()):
    print("Remapping checkpoint keys...")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.features.'):
            new_key = key.replace('backbone.features.', 'features.', 1)
            new_state_dict[new_key] = value
        elif key.startswith('backbone.'):
            new_key = key.replace('backbone.', '', 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

# Load with strict=False
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"⚠️ Missing keys: {len(missing)}")
if unexpected:
    print(f"⚠️ Unexpected keys: {len(unexpected)}")

model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test with actual images
print("\n" + "="*60)
print("TESTING MODEL WITH ACTUAL IMAGES")
print("="*60)

# Find test images
data_dir = Path("data/chest_xray_final")
if not data_dir.exists():
    data_dir = Path("data/chest_xray")
if not data_dir.exists():
    data_dir = Path("data/chest_xray_cleaned")

test_normal = None
test_pneumonia = None

if data_dir.exists():
    normal_dir = data_dir / "test" / "NORMAL"
    pneumonia_dir = data_dir / "test" / "PNEUMONIA"
    
    if normal_dir.exists():
        normal_files = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
        if normal_files:
            test_normal = normal_files[0]
    
    if pneumonia_dir.exists():
        pneumonia_files = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg"))
        if pneumonia_files:
            test_pneumonia = pneumonia_files[0]

# Test function
def test_image(image_path, expected_class):
    if not image_path or not image_path.exists():
        return None
    
    print(f"\n📸 Testing: {image_path.name}")
    print(f"   Expected: {expected_class}")
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    raw_outputs = outputs[0].numpy()
    probs = probabilities[0].numpy()
    
    # Test both class orders
    class_names_order1 = ["NORMAL", "PNEUMONIA"]  # Standard order
    class_names_order2 = ["PNEUMONIA", "NORMAL"]  # Inverted order
    
    pred_idx1 = probs.argmax()
    pred_idx2 = probs.argmax()  # Same index
    
    pred1 = class_names_order1[pred_idx1]
    pred2 = class_names_order2[pred_idx2]
    
    print(f"   Raw outputs: [idx0={raw_outputs[0]:.4f}, idx1={raw_outputs[1]:.4f}]")
    print(f"   Probabilities: [idx0={probs[0]:.4f}, idx1={probs[1]:.4f}]")
    print(f"   Prediction (order1): {pred1} (idx={pred_idx1}, conf={probs[pred_idx1]:.4f})")
    print(f"   Prediction (order2): {pred2} (idx={pred_idx2}, conf={probs[pred_idx2]:.4f})")
    
    # Determine correct mapping
    if pred_idx1 == 0:
        print(f"   → Model thinks idx0 is higher → {class_names_order1[0]}")
    else:
        print(f"   → Model thinks idx1 is higher → {class_names_order1[1]}")
    
    return {
        'raw_outputs': raw_outputs,
        'probabilities': probs,
        'pred_idx': pred_idx1,
        'expected': expected_class
    }

# Run tests
results = []

if test_normal:
    result = test_image(test_normal, "NORMAL")
    if result:
        results.append(("NORMAL", result))
else:
    print("\n⚠️ No NORMAL test image found")

if test_pneumonia:
    result = test_image(test_pneumonia, "PNEUMONIA")
    if result:
        results.append(("PNEUMONIA", result))
else:
    print("\n⚠️ No PNEUMONIA test image found")

# Summary
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if len(results) == 2:
    normal_result = results[0][1]
    pneumonia_result = results[1][1]
    
    print(f"\nNORMAL image:")
    print(f"  Model output idx0={normal_result['raw_outputs'][0]:.4f}, idx1={normal_result['raw_outputs'][1]:.4f}")
    print(f"  Higher index: {normal_result['pred_idx']}")
    
    print(f"\nPNEUMONIA image:")
    print(f"  Model output idx0={pneumonia_result['raw_outputs'][0]:.4f}, idx1={pneumonia_result['raw_outputs'][1]:.4f}")
    print(f"  Higher index: {pneumonia_result['pred_idx']}")
    
    # Determine correct class mapping
    if normal_result['pred_idx'] == pneumonia_result['pred_idx']:
        print("\n❌ PROBLEM: Model gives same prediction for both classes!")
        print("   This suggests the model isn't working correctly.")
    else:
        if normal_result['pred_idx'] == 0:
            print("\n✅ Model mapping: idx0 = NORMAL, idx1 = PNEUMONIA")
            print("   Use: class_names = ['NORMAL', 'PNEUMONIA']")
        else:
            print("\n✅ Model mapping: idx0 = PNEUMONIA, idx1 = NORMAL")
            print("   Use: class_names = ['PNEUMONIA', 'NORMAL']")
else:
    print("\n⚠️ Need both NORMAL and PNEUMONIA images to diagnose")

print("\n" + "="*60)

