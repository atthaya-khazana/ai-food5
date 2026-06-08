"""
Flask App - ResNet34 Food Classification + Auth + History with Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import base64
import traceback
from datetime import datetime, timezone, timedelta
WIB = timezone(timedelta(hours=7))
from functools import wraps
from pathlib import Path

from flask import Flask, request, jsonify, render_template, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMGSZ = 224
CLASS_NAMES = ['burger', 'donut', 'fries', 'pizza', 'taco']
MODEL_PATH = "resnet_model.pth"
CONFIDENCE_THRESHOLD = 0.55

CALORIE_TABLE = {
    'burger': 280,
    'donut':  413,
    'pizza':  220,
    'fries':  264,
    'taco':   211,
}

# ============ APP SETUP ============
app = Flask(__name__)
app.secret_key = 'CHANGE_THIS_TO_A_RANDOM_SECRET_KEY_IN_PRODUCTION'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db = SQLAlchemy(app)
model = None


# ============ MODELS ============

class User(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(120), nullable=False)
    email        = db.Column(db.String(200), unique=True, nullable=False)
    phone        = db.Column(db.String(30), nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans        = db.relationship('ScanHistory', backref='user', lazy=True,
                                   cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id':    self.id,
            'name':  self.name,
            'email': self.email,
            'phone': self.phone or '',
        }


class ScanHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    food_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    calorie    = db.Column(db.Integer)
    image_b64  = db.Column(db.Text, nullable=True)   # base64 thumbnail
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':         self.id,
            'food':       self.food_class,
            'confidence': self.confidence,
            'kcal':       self.calorie,
            'image':      self.image_b64,
            'timestamp':  (self.created_at + timedelta(hours=7)).strftime('%d %b %Y, %H:%M'),
        }


with app.app_context():
    db.create_all()


# ============ AUTH HELPERS ============

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        return f(*args, **kwargs)
    return decorated


def current_user():
    uid = session.get('user_id')
    if uid:
        return db.session.get(User, uid)
    return None


# ============ MODEL LOADING ============

def make_thumbnail_b64(image_bytes: bytes, size: int = 120) -> str:
    """Resize image to thumbnail and return base64 string."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def load_model():
    global model
    try:
        print("[MODEL] Initializing ResNet34...")
        model = models.resnet34(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_CLASSES)
        )
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        print(f"[MODEL] ✓ Loaded | Device: {DEVICE} | Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return False


def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMGSZ * 1.143)),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict_image(image_bytes: bytes, return_all_probs: bool = False) -> dict:
    if model is None:
        return {'error': 'Model not loaded'}
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = get_transform()(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        pred_idx = int(probs.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'status':     'REJECTED',
                'confidence': confidence,
                'threshold':  CONFIDENCE_THRESHOLD,
                'message':    ('The uploaded image does not match any supported food. '
                               'Please upload a Burger, Donut, Pizza, Fries, or Taco.'),
            }

        result = {
            'status':        'SUCCESS',
            'class':         pred_class,
            'confidence':    confidence,
            'predicted_idx': pred_idx,
        }
        if return_all_probs:
            result['all_probabilities'] = {
                CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
            }
        return result
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


# ============ AUTH ROUTES ============

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    data = request.get_json(force=True)
    name     = (data.get('name') or '').strip()
    email    = (data.get('email') or '').strip().lower()
    phone    = (data.get('phone') or '').strip()
    password = data.get('password') or ''

    if not name or not email or not password:
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    user = User(name=name, email=email, phone=phone)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id
    return jsonify({'message': 'Account created', 'user': user.to_dict()}), 201


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data  = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    session['user_id'] = user.id
    return jsonify({'message': 'Logged in', 'user': user.to_dict()})


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': 'Logged out'})


@app.route('/api/auth/me', methods=['GET'])
def api_me():
    user = current_user()
    if not user:
        return jsonify({'user': None})
    return jsonify({'user': user.to_dict()})


@app.route('/api/auth/update', methods=['PUT'])
@login_required
def api_update_profile():
    user = current_user()
    data = request.get_json(force=True)
    name  = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    phone = (data.get('phone') or '').strip()

    if not name or not email:
        return jsonify({'error': 'Name and email are required'}), 400

    # Check email collision with another user
    existing = User.query.filter_by(email=email).first()
    if existing and existing.id != user.id:
        return jsonify({'error': 'Email already in use'}), 409

    user.name  = name
    user.email = email
    user.phone = phone
    db.session.commit()
    return jsonify({'message': 'Profile updated', 'user': user.to_dict()})

@app.route('/api/auth/change-password', methods=['PUT'])
@login_required
def api_change_password():
    user = current_user()
    data = request.get_json(force=True)
    old_password = data.get('old_password') or ''
    new_password = data.get('new_password') or ''

    if not user.check_password(old_password):
        return jsonify({'error': 'Current password is incorrect'}), 400
    if len(new_password) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400

    user.set_password(new_password)
    db.session.commit()
    return jsonify({'message': 'Password changed successfully'})

# ============ PREDICT ROUTES ============

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'ok', 'model_ready': model is not None, 'device': str(DEVICE)})


@app.route('/api/debug', methods=['GET'])
def api_debug():
    return jsonify({
        'status': 'ok', 'device': str(DEVICE), 'model_loaded': model is not None,
        'num_classes': NUM_CLASSES, 'class_names': CLASS_NAMES,
        'confidence_threshold': CONFIDENCE_THRESHOLD, 'calorie_table': CALORIE_TABLE,
    })


@app.route('/api/predict-debug', methods=['POST'])
def api_predict_debug():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_bytes = request.files['image'].read()
    result = predict_image(image_bytes, return_all_probs=True)

    if 'error' in result:
        return jsonify(result), 400

    result['timestamp'] = datetime.now().isoformat()

    if result['status'] == 'REJECTED':
        return jsonify(result)

    result['calorie'] = CALORIE_TABLE.get(result['class'])

    # Save to history (linked to user if logged in)
    print("WAKTU SEKARANG:", datetime.now(WIB))
    print("WAKTU UTC:", datetime.utcnow())
    thumb = make_thumbnail_b64(image_bytes)
    entry = ScanHistory(
        user_id    = session.get('user_id'),
        food_class = result['class'],
        confidence = result['confidence'],
        calorie    = result['calorie'],
        image_b64  = thumb,
    )
    db.session.add(entry)
    db.session.commit()
    result['history_id'] = entry.id

    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_bytes = request.files['image'].read()
    result = predict_image(image_bytes)

    if 'error' in result:
        return jsonify(result), 400

    result['timestamp'] = datetime.now().isoformat()

    if result['status'] == 'REJECTED':
        return jsonify(result)

    result['calorie'] = CALORIE_TABLE.get(result['class'])

    thumb = make_thumbnail_b64(image_bytes)
    entry = ScanHistory(
        user_id    = session.get('user_id'),
        food_class = result['class'],
        confidence = result['confidence'],
        calorie    = result['calorie'],
        image_b64  = thumb,
    )
    db.session.add(entry)
    db.session.commit()
    result['history_id'] = entry.id

    return jsonify(result)


# ============ HISTORY ROUTES ============

@app.route('/api/history', methods=['GET'])
def api_get_history():
    uid = session.get('user_id')
    if uid:
        rows = ScanHistory.query.filter_by(user_id=uid)\
                  .order_by(ScanHistory.created_at.desc()).all()
    else:
        # Guest: return empty (or you could use a session key)
        rows = []
    return jsonify([r.to_dict() for r in rows])


@app.route('/api/history/<int:entry_id>', methods=['DELETE'])
def api_delete_history(entry_id):
    row = db.session.get(ScanHistory, entry_id)
    if not row:
        return jsonify({'error': 'Not found'}), 404
    # Only owner can delete
    if row.user_id and row.user_id != session.get('user_id'):
        return jsonify({'error': 'Forbidden'}), 403
    db.session.delete(row)
    db.session.commit()
    return jsonify({'deleted': entry_id})


@app.route('/api/history', methods=['DELETE'])
def api_clear_history():
    uid = session.get('user_id')
    if uid:
        ScanHistory.query.filter_by(user_id=uid).delete()
    db.session.commit()
    return jsonify({'cleared': True})


# ============ ERROR HANDLERS ============

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


# ============ STARTUP ============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍔  ResNet34 Food Classification — Auth Edition")
    print("="*60)
    success = load_model()
    if not success:
        print("⚠️  Model not loaded — /api/predict will fail until model file is present.")
    app.run(debug=True, host='0.0.0.0', port=5000)
