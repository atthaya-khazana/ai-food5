"""
Debug Helper Script - Untuk test model, preprocessing, dan class order
=====================================================================

Usage:
    python debug_helper.py  # Run all checks
    python debug_helper.py --test-image burger.jpg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet_model.pth"
NUM_CLASSES = 5
IMGSZ = 224

CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']

# ============ COLORS ============
class Colors:
    OK = '\033[92m'      # Green
    WARN = '\033[93m'    # Yellow
    ERROR = '\033[91m'   # Red
    INFO = '\033[94m'    # Blue
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.INFO}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.END}\n")

def print_ok(text):
    print(f"{Colors.OK}✓ {text}{Colors.END}")

def print_warn(text):
    print(f"{Colors.WARN}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.ERROR}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.INFO}ℹ {text}{Colors.END}")

# ============ CHECK 1: Model File ============
def check_model_file():
    print_header("CHECK 1: Model File")
    
    path = Path(MODEL_PATH)
    if not path.exists():
        print_error(f"Model file tidak ditemukan: {MODEL_PATH}")
        return False
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print_ok(f"Model file ditemukan: {MODEL_PATH}")
    print_info(f"File size: {size_mb:.2f} MB")
    
    return True

# ============ CHECK 2: State Dict ============
def check_state_dict():
    print_header("CHECK 2: State Dict Keys")
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        print_ok(f"State dict loaded successfully")
        print_info(f"Total parameters: {len(state_dict)}")
        
        # Print first 5 keys
        print(f"\nFirst 5 keys in state_dict:")
        for key in list(state_dict.keys())[:5]:
            shape = state_dict[key].shape
            print(f"  {key:40s} {shape}")
        
        # Check FC layer
        fc_key = [k for k in state_dict.keys() if 'fc' in k]
        if fc_key:
            print(f"\nFC layer keys:")
            for key in fc_key:
                shape = state_dict[key].shape
                print(f"  {key:40s} {shape}")
        
        return state_dict
        
    except Exception as e:
        print_error(f"Failed to load state dict: {e}")
        return None

# ============ CHECK 3: Model Architecture ============
def check_model_architecture(state_dict):
    print_header("CHECK 3: Model Architecture")
    
    try:
        model = models.resnet34(weights=None)
        print_ok("ResNet34 initialized")
        
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace FC
        in_features = model.fc.in_features
        print_info(f"FC input features: {in_features}")
        
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_CLASSES)
        )
        print_ok(f"FC layer replaced with Sequential (Dropout + Linear)")
        
        # Try to load
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        print_ok(f"State dict loaded into model successfully")
        print_info(f"Device: {DEVICE}")
        
        return model
        
    except Exception as e:
        print_error(f"Failed to build/load model: {e}")
        return None

# ============ CHECK 4: Preprocessing ============
def check_preprocessing():
    print_header("CHECK 4: Preprocessing Consistency")
    
    print_info("Training transform:")
    print(f"  Resize({int(IMGSZ * 1.143)})")
    print(f"  CenterCrop({IMGSZ})")
    print(f"  ToTensor()")
    print(f"  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    
    transform = transforms.Compose([
        transforms.Resize(int(IMGSZ * 1.143)),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print_ok("Transform initialized")
    
    return transform

# ============ CHECK 5: Class Names ============
def check_class_names():
    print_header("CHECK 5: Class Names Order")
    
    print_info("Expected order (alphabetical from ImageFolder):")
    print(f"  {' → '.join(CLASS_NAMES)}")
    
    expected = ['burger', 'donut', 'fries', 'pizza', 'taco']
    
    if CLASS_NAMES == expected:
        print_ok("CLASS_NAMES order is correct (alphabetical)")
    else:
        print_warn(f"CLASS_NAMES might be in wrong order!")
        print(f"  Expected: {expected}")
        print(f"  Got: {CLASS_NAMES}")
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Index {i}: {name}")

# ============ CHECK 6: Test Image ============
def test_image(model, transform, image_path):
    print_header(f"CHECK 6: Test Inference on {image_path}")
    
    if not Path(image_path).exists():
        print_error(f"Image not found: {image_path}")
        return
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        print_ok(f"Image loaded: {img.size} (W x H)")
        
        # Transform
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        print_ok(f"Transform applied: tensor shape {tensor.shape}")
        
        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        print_ok(f"Inference completed")
        print(f"\n{Colors.BOLD}Prediction Result:{Colors.END}")
        print(f"  Class: {pred_class}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        
        print(f"\n{Colors.BOLD}All Class Probabilities:{Colors.END}")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs.tolist())):
            marker = " ← SELECTED" if i == pred_idx else ""
            bar_length = int(prob * 30)
            bar = "█" * bar_length
            print(f"  {name:10s} {prob:6.4f} {bar:30s}{marker}")
        
        # Save result
        result = {
            'image': image_path,
            'timestamp': datetime.now().isoformat(),
            'prediction': pred_class,
            'confidence': confidence,
            'all_probabilities': {
                CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            }
        }
        
        result_file = Path(image_path).stem + '_debug_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print_ok(f"Result saved to {result_file}")
        
    except Exception as e:
        print_error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

# ============ DUMMY IMAGE TEST ============
def test_dummy_image(model):
    print_header("CHECK 7: Dummy Image Test")
    
    print_info("Creating dummy image (all pixels = 0.5)")
    
    # Dummy tensor (semua pixel 0.5 = neutral)
    dummy_tensor = torch.ones(1, 3, 224, 224) * 0.5
    dummy_tensor = dummy_tensor.to(DEVICE)
    
    with torch.no_grad():
        logits = model(dummy_tensor)
        probs = F.softmax(logits, dim=1)[0]
    
    pred_idx = int(probs.argmax())
    
    print(f"\nDummy image prediction:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs.tolist())):
        marker = " ← SELECTED" if i == pred_idx else ""
        print(f"  {name:10s}: {prob:.4f}{marker}")
    
    print_info("Dummy image should have similar probabilities for all classes")
    print_info("If one class has much higher prob, it might indicate overfitting")

# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description='Debug ResNet34 Food Classification')
    parser.add_argument('--test-image', type=str, help='Test image path')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.INFO}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     ResNet34 Food Classification - Debug Helper            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    # Check 1
    if not check_model_file():
        return
    
    # Check 2
    state_dict = check_state_dict()
    if state_dict is None:
        return
    
    # Check 3
    model = check_model_architecture(state_dict)
    if model is None:
        return
    
    # Check 4
    transform = check_preprocessing()
    
    # Check 5
    check_class_names()
    
    # Check 6
    if args.test_image:
        test_image(model, transform, args.test_image)
    
    # Check 7
    test_dummy_image(model)
    
    # Summary
    print_header("Summary")
    print_ok("All checks completed!")
    print_info("If all checks passed, model is ready for Flask deployment")
    print_info("Next step: python app.py")

if __name__ == '__main__':
    main()
