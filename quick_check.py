#!/usr/bin/env python3
"""
Quick Checklist untuk ResNet34 Debugging
==========================================

Ini akan otomatis mengecek semua hal penting dan memberikan rekomendasi, dengan menjalankan python quick_check.py
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import sys

def check_item(condition, success_msg, failure_msg):
    """Helper untuk print check result"""
    if condition:
        print(f"  ✓ {success_msg}")
        return True
    else:
        print(f"  ✗ {failure_msg}")
        return False

def main():
    print("\n" + "="*70)
    print("🍔 ResNet34 Food Classification - Quick Check")
    print("="*70 + "\n")
    
    all_passed = True
    
    # CHECK 1: Model File
    print("📋 CHECK 1: Model File")
    model_exists = Path("resnet_model.pth").exists()
    all_passed &= check_item(
        model_exists,
        "Model file resnet_model.pth ditemukan",
        "Model file tidak ditemukan - pastikan resnet_model.pth ada di directory ini"
    )
    
    if not model_exists:
        print("\n  ⚠️  STOP: Model file required!\n")
        return False
    
    # CHECK 2: Load State Dict
    print("\n📋 CHECK 2: Load Model Checkpoint")
    try:
        state_dict = torch.load("resnet_model.pth", map_location='cpu')
        all_passed &= check_item(
            True,
            f"Model checkpoint loaded ({len(state_dict)} keys)",
            "Failed to load checkpoint"
        )
    except Exception as e:
        check_item(False, "", f"Error loading checkpoint: {e}")
        return False
    
    # CHECK 3: Build Architecture
    print("\n📋 CHECK 3: Model Architecture")
    try:
        model = models.resnet34(weights=None)
        
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        
        # FC layer
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 5)
        )
        
        # Load state
        model.load_state_dict(state_dict)
        model.eval()
        
        all_passed &= check_item(
            True,
            "Model architecture built and weights loaded successfully",
            "Failed to build architecture"
        )
    except Exception as e:
        check_item(False, "", f"Error building model: {e}")
        return False
    
    # CHECK 4: Image Preprocessing
    print("\n📋 CHECK 4: Preprocessing Configuration")
    
    IMGSZ = 224
    expected_resize = int(IMGSZ * 1.143)  # Should be 256
    
    all_passed &= check_item(
        expected_resize == 256,
        f"Image size resize value = {expected_resize} (correct for IMGSZ=224)",
        f"Resize value calculation wrong: {expected_resize}"
    )
    
    transform = transforms.Compose([
        transforms.Resize(expected_resize),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    all_passed &= check_item(
        True,
        "Preprocessing transform created (Resize→CenterCrop→Normalize)",
        "Failed to create transform"
    )
    
    # CHECK 5: Class Names
    print("\n📋 CHECK 5: Class Names Order")
    
    CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']
    EXPECTED = ['burger', 'donut', 'fries', 'pizza', 'taco']
    
    is_correct_order = CLASS_NAMES == EXPECTED
    all_passed &= check_item(
        is_correct_order,
        f"CLASS_NAMES in correct alphabetical order",
        f"CLASS_NAMES order might be wrong!\n    Expected: {EXPECTED}\n    Got: {CLASS_NAMES}"
    )
    
    print("\n  Class to Index mapping:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"    {idx}: {name}")
    
    # CHECK 6: Test Inference
    print("\n📋 CHECK 6: Model Inference")
    
    try:
        dummy = torch.ones(1, 3, 224, 224) * 0.5
        with torch.no_grad():
            out = model(dummy)
        
        all_passed &= check_item(
            out.shape == (1, 5),
            f"Model output shape correct: {out.shape} (expected: (1, 5))",
            f"Model output shape wrong: {out.shape}"
        )
    except Exception as e:
        check_item(False, "", f"Error during inference: {e}")
        all_passed = False
    
    # SUMMARY
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for deployment!\n")
        print("Next steps:")
        print("  1. python app.py              (start Flask server)")
        print("  2. Open http://localhost:5000 (in browser)")
        print("  3. Upload food image          (test predictions)")
        print()
        return True
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above\n")
        print("Debugging steps:")
        print("  1. Check all error messages above")
        print("  2. Run: python debug_helper.py --test-image burger.jpg")
        print("  3. Read: DEBUGGING_GUIDE.md")
        print()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
