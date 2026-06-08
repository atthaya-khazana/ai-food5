# 🍔 Deep Learning-Based Fast Food Detection for Web-Based Calorie Estimation

Complete Flask application dengan HTML frontend + PyTorch ResNet34 model untuk food classification dan calorie estimation.

## 📂 File Structure

```
resnet_food_project/
├── app.py                  # Core Flask application for food classification and calorie estimation
├── debug_helper.py         # Utility script for model verification and debugging
├── quick_check.py          # Script for quick functionality testing
├── requirements.txt        # Required Python packages
├── resnet_model.pth        # Pre-trained ResNet34 model weights
├── README.md               # Project documentation
├── templates/
│   └── index.html          # Web interface for image upload and prediction results
└── instance/               # Application-specific configuration files
    └── history.db         
```
This project is a web-based application that uses a deep learning model (ResNet34) to classify fast food images and estimate their calorie content. The system recognizes five food categories: Burger, Donut, Fries, Pizza, and Taco, and provides calorie estimation based on predefined nutritional references.

## 🚀 Quick Start

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Model

Make sure `resnet_model.pth` is in the same directory as `app.py`.

```bash
ls -la resnet_model.pth  # Check file exists
```

### Step 3: Run Debug Helper (RECOMMENDED FIRST)

**This checks if model and preprocessing are correct:**

```bash
# Check all configurations
python debug_helper.py

# Test with a specific image
python debug_helper.py --test-image burger.jpg
```

**Output should show:**
- ✓ Model file found
- ✓ State dict loaded
- ✓ Model architecture correct
- ✓ Preprocessing config correct
- ✓ Class names order correct

### Step 4: Run Flask App

```bash
python app.py
```

Output:
```
============================================================
🍔 ResNet34 Food Classification Flask App
============================================================

[MODEL] Initializing ResNet34...
[MODEL] Loading weights dari resnet_model.pth...
[MODEL] ✓ Model loaded successfully
[MODEL] Device: cpu
[MODEL] Classes (5): ['burger', 'donut', 'fries', 'pizza', 'taco']

📋 Available Endpoints:
   GET  /                    -> Frontend HTML
   GET  /api/health          -> Health check
   GET  /api/debug           -> Model info & preprocessing
   POST /api/predict-debug   -> Predict + all class probabilities
   POST /api/predict         -> Predict (production)

============================================================

 * Running on http://0.0.0.0:5000
```

### Step 5: Open Browser

Navigate to: **http://localhost:5000**

Upload a food image and see predictions!

---

## 🔍 API Endpoints

### GET `/`
Frontend HTML interface

### GET `/api/health`
Health check endpoint
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "ok",
  "model_ready": true,
  "device": "cpu"
}
```

### GET `/api/debug`
Get model configuration and preprocessing details
```bash
curl http://localhost:5000/api/debug
```

Response:
```json
{
  "status": "ok",
  "device": "cpu",
  "model_path": "resnet_model.pth",
  "model_loaded": true,
  "num_classes": 5,
  "class_names": ["burger", "donut", "fries", "pizza", "taco"],
  "image_size": 224,
  "preprocessing": {
    "resize": 256,
    "center_crop": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

### POST `/api/predict-debug`
Predict with full debugging information (all class probabilities)

```bash
curl -F "image=@burger.jpg" http://localhost:5000/api/predict-debug
```

Response:
```json
{
  "class": "burger",
  "confidence": 0.962,
  "predicted_idx": 0,
  "image_shape": [800, 600],
  "tensor_shape": [1, 3, 224, 224],
  "all_probabilities": {
    "burger": 0.962,
    "donut": 0.01,
    "fries": 0.005,
    "pizza": 0.015,
    "taco": 0.008
  },
  "calorie": 295
}
```

### POST `/api/predict`
Production prediction endpoint (only returns main prediction)

```bash
curl -F "image=@burger.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "class": "burger",
  "confidence": 0.962,
  "predicted_idx": 0,
  "calorie": 295
}
```

---

## 🛠️ Configuration

Edit `app.py` to change:

```python
# Model config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMGSZ = 224
CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']  # ⚠️ MUST be alphabetical!
MODEL_PATH = "resnet_model.pth"

# Calorie lookup table (from FNDDS dataset)
CALORIE_TABLE = {
    'burger': 280.129,
    'donut': 412.538,
    'pizza': 264.022,
    'fries': 220.250,
    'taco': 210.815
}
```

---

## ⚠️ Common Issues

### Issue 1: "Model file tidak ditemukan"

**Solution:**
```bash
# Make sure resnet_model.pth exists
ls -la resnet_model.pth

# If missing, train model first or download from Google Drive
# (depends on your training code)
```

### Issue 2: "Predictions are always wrong"

**Solution:** Run debugging steps:

```bash
# 1. Check model and preprocessing
python debug_helper.py

# 2. Test with known image
python debug_helper.py --test-image burger.jpg

```

Most common cause: **CLASS_NAMES order is wrong**

Saat training dengan ImageFolder, PyTorch otomatis menggunakan alphabetical order:
```
✅ CORRECT:  ['burger', 'donut', 'fries', 'pizza', 'taco']
❌ WRONG:    ['taco', 'pizza', 'fries', 'donut', 'burger']
```

### Issue 3: "CUDA out of memory"

**Solution:**
```python
# In app.py, change to CPU:
DEVICE = torch.device("cpu")  # Instead of 'cuda'
```

### Issue 4: "Slow predictions"

**Solution:**
```python
# If using CPU, predictions might be slow
# Consider:
# 1. Use GPU (NVIDIA CUDA compatible GPU)
# 2. Reduce image size (change IMGSZ)
# 3. Use lighter model (MobileNetV3 instead of ResNet34)
```

---

## 📊 Frontend Features

- **Real-time predictions** - Upload image, get instant results
- **Confidence display** - Shows prediction confidence percentage
- **Calorie lookup** - Automatic calorie estimation from FNDDS dataset
- **All class probabilities** - Debug mode shows confidence for all 5 classes
- **Drag-and-drop** - Upload images easily
- **Responsive design** - Works on desktop and mobile

---

## 🔬 Debug Mode

The HTML frontend includes debug mode that shows all class probabilities:

```javascript
const DEBUG_MODE = true;  // In index.html
```

When enabled, you'll see:
```
All Class Probabilities
burger  : 96.2% ← SELECTED
donut   : 1.0%
fries   : 0.5%
pizza   : 1.5%
taco    : 0.8%
```

This helps verify if predictions are correct.

---

## 📚 Training Model Reference

The model was trained with:
- **Architecture**: ResNet34 (pretrained on ImageNet, backbone frozen)
- **FC Layer**: Dropout(0.3) + Linear(512 → 5)
- **Dataset**: 5 food classes (burger, donut, fries, pizza, taco)
- **Validation Accuracy**: ~93-94%
- **Image Size**: 224x224
- **Preprocessing**: ImageNet normalization

---

## 🚀 Deployment

### Local Testing
```bash
python app.py
# Access at http://localhost:5000
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
docker build -t food-classifier .
docker run -p 5000:5000 food-classifier
```

---

## 📈 Performance Tips

1. **GPU acceleration** - Use CUDA if available
   ```python
   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Model quantization** - Convert model to int8 for faster inference
   ```python
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

3. **Batch processing** - Process multiple images at once for efficiency

4. **Caching** - Cache predictions for identical images

---

## 📝 Training & Validation Data

This model was trained on food images from:
- **Dataset**: Fast food images (Burger, Donut, Pizza, Fries, Taco)
- **Dataset Source**: Roboflow Fast Food Classification Dataset
- **Classes**: 5 (balanced)
- **Split**: 70% train, 15% val, 15% test

Validation results:
```
burger → burger (96.2%)
donut  → donut  (90.3%)
fries  → fries  (92.7%)
pizza  → pizza  (93.9%)
taco   → taco   (91.5%)
```

---
## 🎯 Features

- Fast food image classification using ResNet34
- Web-based user interface built with Flask
- Real-time prediction and confidence score display
- Calorie estimation using USDA FNDDS nutritional data
- Support for five food categories: Burger, Donut, Fries, Pizza, and Taco

---
## 🤝 Support

If predictions are wrong:

1. **Check CLASS_NAMES order** (most common issue)
   ```python
   CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']
   ```

2. **Check preprocessing matches training**
   - Resize: 256
   - CenterCrop: 224
   - Normalize: ImageNet standard

3. **Test with debug helper**
   ```bash
   python debug_helper.py --test-image burger.jpg
   ```

---

## 📄 License & Attribution

- Model Architecture: ResNet34 (TorchVision)
- Image Dataset: Roboflow Fast Food Classification Dataset
- Calorie Reference: USDA FNDDS Dataset
- Frontend: HTML, CSS, JavaScript

---

**Last Updated**: 2026
**Model Version**: v1.0
**Python Version**: 3.8+
