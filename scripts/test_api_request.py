import requests
import json
import base64
from PIL import Image
import io
import numpy as np

def test_api():
    # Create a dummy chest X-ray image (380x380 grayscale for EfficientNet-B4)
    dummy_image = np.random.randint(0, 255, (380, 380), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image, mode='L')
    
    # Convert to RGB (3 channels) as expected by the model
    pil_image = pil_image.convert('RGB')
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Test the API
    url = "http://localhost:5000/predict"
    data = {"image": img_base64}
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()