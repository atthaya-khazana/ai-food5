# 🔍 DEBUGGING GUIDE: ResNet34 Food Classification

## RINGKASAN MASALAH

Model Anda prediksi salah (burger → taco). Ini **BUKAN** masalah akurasi model saat training,
tapi masalah **MAPPING atau PREPROCESSING** antara training dan inference.

---

## 📋 JAWABAN 7 PERTANYAAN CASEY

### 1. **Apakah urutan CLASS_NAMES mungkin salah?**
   ✅ **INI ADALAH MASALAH PALING UMUM**
   
   Saat training dengan `ImageFolder`, PyTorch otomatis membuat `class_to_idx` berdasarkan **urutan alfabetis** folder:
   ```
   burger  → index 0
   donut   → index 1
   fries   → index 2
   pizza   → index 3
   taco    → index 4
   ```
   
   Jika CLASS_NAMES di Flask berbeda urutan, prediksi akan SALAH.
   
   **Solusi:** Pastikan CLASS_NAMES di Flask **persis sama** dengan urutan folder saat training.

---

### 2. **Apakah class_to_idx dari ImageFolder saat training berbeda?**
   ✅ **MUNGKIN**
   
   Jika saat training Anda print `dataset.class_to_idx`, urutan folder akan jelas.
   Contoh, jika folder strukturnya:
   ```
   data/
     burger/     → idx 0
     donut/      → idx 1
     fries/      → idx 2
     pizza/      → idx 3
     taco/       → idx 4
   ```
   
   Tapi di Flask Anda tuliskan:
   ```python
   CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']  # ✅ BENAR
   CLASS_NAMES = ['taco', 'pizza', 'fries', 'donut', 'burger']  # ❌ SALAH → prediksi terbalik
   ```

---

### 3. **Apakah preprocessing inferensi tidak sama dengan training?**
   ✅ **INI JUGA MASALAH UMUM**
   
   **Training:**
   ```python
   val_transform = T.Compose([
       T.Resize(int(IMGSZ * 1.143)),  # IMGSZ = 224 → Resize(256)
       T.CenterCrop(IMGSZ),            # CenterCrop(224)
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```
   
   **Flask (yang Anda berikan):**
   ```python
   transform = transforms.Compose([
       transforms.Resize(256),           # ✅ SAMA
       transforms.CenterCrop(224),       # ✅ SAMA
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```
   
   Jika di Flask Anda gunakan `Resize(224)` langsung tanpa `int(IMGSZ * 1.143)`, hasilnya BERBEDA.
   **Karena CenterCrop akan memotong area berbeda.**

---

### 4. **Apakah model yang tersimpan bukan model terbaik?**
   ✅ **MUNGKIN**
   
   Jika Anda simpan model di epoch 10 (loss masih turun), tapi test data akurat 93-94%,
   kemungkinan model tersimpan bukan yang terbaik.
   
   **Solusi:**
   - Simpan model hanya saat validation accuracy meningkat (checkpoint)
   - Atau simpan setiap epoch, lalu test dengan berbagai model untuk pastikan

---

### 5. **Bagaimana cara mengecek probabilitas semua kelas saat inferensi?**
   ✅ **GUNAKAN /api/predict-debug**
   
   Endpoint ini mengembalikan:
   ```json
   {
     "class": "burger",
     "confidence": 0.962,
     "all_probabilities": {
       "burger": 0.962,
       "donut": 0.01,
       "fries": 0.005,
       "pizza": 0.015,
       "taco": 0.008
     }
   }
   ```
   
   Jika Anda upload gambar burger tapi hasil:
   ```json
   {
     "class": "taco",
     "confidence": 0.85,
     "all_probabilities": {
       "burger": 0.05,        ← burger hanya 5%
       "donut": 0.03,
       "fries": 0.02,
       "pizza": 0.05,
       "taco": 0.85           ← taco 85% (SALAH)
     }
   }
   ```
   
   Ada 2 kemungkinan:
   - **Model salah diprediksi** (taco similarity tinggi dengan burger di training data)
   - **Preprocessing berbeda** (saat training preprocessing berbeda)
   - **CLASS_NAMES mismatch** (burger disimpan dengan label taco)

---

### 6. **Bagaimana cara memastikan mapping label dan model benar?**
   ✅ **GUNAKAN VALIDATION SET SAAT TRAINING**
   
   Saat training, pastikan Anda save hasil prediction di validation set:
   ```python
   # Saat validation
   val_results = []
   for image, label in val_loader:
       with torch.no_grad():
           outputs = model(image)
           probs = F.softmax(outputs, dim=1)
           preds = probs.argmax(dim=1)
       
       for i, (pred_idx, true_idx) in enumerate(zip(preds, label)):
           val_results.append({
               'image_name': val_images[idx],
               'true_class': CLASS_NAMES[true_idx],
               'pred_class': CLASS_NAMES[pred_idx],
               'confidence': float(probs[i, pred_idx]),
               'all_probs': {CLASS_NAMES[j]: float(probs[i, j]) for j in range(NUM_CLASSES)}
           })
   
   # Save ke JSON untuk referensi
   import json
   with open('validation_results.json', 'w') as f:
       json.dump(val_results, f, indent=2)
   ```
   
   Lalu bandingkan dengan hasil Flask:
   - Apakah prediksi sama?
   - Apakah confidence sama?
   - Apakah all_probabilities sama?

---

### 7. **Kode apa yang harus saya tambahkan untuk debugging?**
   ✅ **LIHAT BAGIAN BERIKUT**

---

## 🛠️ LANGKAH DEBUGGING SISTEMATIS

### STEP 1: Check Model Loading
```bash
python -c "
import torch
model_path = 'resnet_model.pth'
state_dict = torch.load(model_path, map_location='cpu')
print('Keys in state_dict:', list(state_dict.keys())[:5])  # Print first 5 keys
print('Total keys:', len(state_dict))

# Cek architecture
from torchvision import models
model = models.resnet34(weights=None)
print('\nModel keys:', list(model.state_dict().keys())[:5])
"
```

**Expected output:** Keys harus match antara model architecture dan saved state_dict.

---

### STEP 2: Check Class Order (PENTING!)

**Opsi A: Lihat training script**
```python
# Jika Anda masih punya training code, print ini:
from torchvision.datasets import ImageFolder
train_dataset = ImageFolder('path/to/train/data')
print('class_to_idx:', train_dataset.class_to_idx)
print('classes:', train_dataset.classes)
```

**Opsi B: Inference dengan dummy image**
```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

DEVICE = torch.device('cpu')
NUM_CLASSES = 5

# Load model
model = models.resnet34(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load('resnet_model.pth', map_location=DEVICE))
model.eval()

# Create dummy image (semua pixel 0.5 = neutral)
dummy_tensor = torch.ones(1, 3, 224, 224) * 0.5
dummy_tensor = dummy_tensor.to(DEVICE)

with torch.no_grad():
    outputs = model(dummy_tensor)
    probs = F.softmax(outputs, dim=1)

print('Probabilities untuk dummy image:', probs)
print('Highest prob class:', probs.argmax())

# CLASS_NAMES yang berbeda akan menghasilkan prediksi berbeda
CLASS_NAMES_1 = ['burger', 'donut', 'fries', 'pizza', 'taco']
CLASS_NAMES_2 = ['taco', 'pizza', 'fries', 'donut', 'burger']

idx = int(probs.argmax())
print(f"CLASS_NAMES_1[{idx}] = {CLASS_NAMES_1[idx]}")  # mungkin 'burger'
print(f"CLASS_NAMES_2[{idx}] = {CLASS_NAMES_2[idx]}")  # mungkin 'taco'
```

---

### STEP 3: Check Preprocessing Consistency

```python
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Image dummy (buat dari file atau generate)
img_path = 'test_burger.jpg'
img = Image.open(img_path).convert('RGB')

# Training transform
train_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.143)),  # 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Flask transform
flask_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_train = train_transform(img)
tensor_flask = flask_transform(img)

# Check apakah sama
diff = torch.abs(tensor_train - tensor_flask).max()
print(f'Max difference between transforms: {diff}')

if diff > 0.01:
    print('⚠️  WARNING: Transforms sangat berbeda!')
else:
    print('✅ Transforms roughly sama')
```

---

### STEP 4: Full Inference Test

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device('cpu')
MODEL_PATH = 'resnet_model.pth'
NUM_CLASSES = 5
IMGSZ = 224

# ⚠️ PERHATIAN: UBAH URUTAN INI SESUAI TRAINING ANDA
CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']

def load_model():
    model = models.resnet34(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMGSZ * 1.143)),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict(image_path):
    model = load_model()
    transform = get_transform()
    
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    
    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    print(f'\n📸 Image: {image_path}')
    print(f'🎯 Prediction: {pred_class} (confidence: {confidence:.2%})')
    print(f'\n📊 All probabilities:')
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs.tolist())):
        marker = ' ← SELECTED' if i == pred_idx else ''
        print(f'   {name:10s}: {prob:.4f} ({prob*100:.1f}%){marker}')

# Test
predict('burger.jpg')
predict('taco.jpg')
```

---

## 🐛 DEBUGGING CHECKLIST

### Sebelum deploy production:

- [ ] **CLASS_NAMES order** = urutan folder saat training ImageFolder
- [ ] **IMGSZ** = size image saat training (default 224)
- [ ] **Resize value** = int(IMGSZ * 1.143) EXACT sama dengan training
- [ ] **Normalize params** = [0.485, 0.456, 0.406] dan [0.229, 0.224, 0.225] (ImageNet default)
- [ ] **Model architecture** = Dropout + Linear FC layer SAMA dengan training
- [ ] **Dropout di inference** = disable (model.eval() sudah handle ini)
- [ ] **Device** = transfer ke correct device saat load
- [ ] **Validation result** = simpan saat training untuk compare dengan Flask

---

## 🚀 NEXT STEPS

1. **Run Flask app** (sudah saya buat di `app.py`):
   ```bash
   python app.py
   ```
   
2. **Open browser** ke `http://localhost:5000`

3. **Upload test image**:
   - Buka `/api/predict-debug` endpoint
   - Lihat semua probabilities
   - Bandingkan dengan hasil training

4. **Jika prediksi masih salah**:
   - Print `validation_results.json` dari training
   - Bandingkan class order
   - Cek preprocessing perbedaan

---

## 📁 FILE STRUCTURE

```
project/
├── app.py                    # Flask app dengan debugging
├── resnet_model.pth         # Model checkpoint
├── templates/
│   └── index.html           # Frontend (real predictions)
└── (optional)
    └── debug_inference.py   # Script untuk testing
```

---

## 🔗 API ENDPOINTS

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Frontend HTML |
| `/api/health` | GET | Health check |
| `/api/debug` | GET | Model info & preprocessing |
| `/api/predict-debug` | POST | Predict + all class probabilities |
| `/api/predict` | POST | Predict (production) |

---

## 💡 TIPS

1. **Test dengan class yang benar**: Upload burger image, lihat apakah burger probability tinggi
2. **Gunakan predict-debug**: Lihat semua probabilities, jangan cuma prediksi utama
3. **Save validation results**: Saat training, simpan prediction setiap image untuk later comparison
4. **Check image quality**: Memastikan test image quality sama dengan training data
5. **GPU/CPU**: Jika training pakai GPU tapi Flask pakai CPU, bisa ada numerical difference kecil

---

## ⚠️ COMMON MISTAKES

❌ **Salah:**
```python
CLASS_NAMES = ['taco', 'pizza', 'fries', 'donut', 'burger']  # Random order
```

✅ **Benar:**
```python
CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']  # Alphabetical
```

---

Semoga sukses! 🍔🎯
