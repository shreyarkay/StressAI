from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)

# ================================
# 1. Load Model and Tokenizer
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("backend", "bert_stress_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ================================
# 2. Route for Landing Page
# ================================
@app.route('/')
def landing():
    return render_template('landing.html')

# ================================
# 3. Route for Dashboard
# ================================
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

# ================================
# 4. Prediction API
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        stress_prob = probs[0][1].item()

        # Stress level logic (same as old)
        if stress_prob < 0.75:
            stress_level = "No Stress"
        else:
            stress_level = "High Stress"

        result = f"{stress_level} ({stress_prob * 100:.2f}% confidence)"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================================
# 5. Run App
# ================================
if __name__ == '__main__':
    app.run(debug=True)
