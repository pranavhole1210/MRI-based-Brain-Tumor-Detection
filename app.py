
# 🔧 Suppress TensorFlow warnings (clean output)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 🌐 Flask (web app)
from flask import Flask, render_template, request, session, send_file

# 📄 PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 📅 Date & time
from datetime import datetime

# 🔢 Numerical operations
import numpy as np

# 🤖 Deep learning (TensorFlow)
import tensorflow as tf

# 🖼️ Image preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 🚀 Create Flask app
app = Flask(__name__)

# 🔐 Secret key (used for storing session data securely)
app.secret_key = 'your-secret-key-12345'


# 📁 Path of trained model file
MODEL_PATH = "mri_brain_tumor_vgg16.h5"

# 🤖 Load trained model (no training, only prediction)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# 🧠 Class labels (same as training dataset)
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 📏 Image size used during training
IMG_SIZE = (128, 128)


# 🏠 Home route (loads main page)
@app.route("/")
def home():
    return render_template("index.html")


# 🔍 Prediction route (when user submits form)
@app.route("/predict", methods=["POST"])
def predict():

    # 📥 Get data from form
    patient_name = request.form.get("patient_name")
    patient_id   = request.form.get("patient_id")
    age          = request.form.get("age")
    doctor       = request.form.get("doctor")

    # 📸 Get uploaded image
    image_file   = request.files.get("image")

    # ❌ If no image uploaded
    if image_file is None or image_file.filename == "":
        prediction = "NO IMAGE"
        confidence = 0.0
        image_path = None

    else:
        # 💾 Save uploaded image
        image_path = "uploaded_image.png"
        image_file.save(image_path)

        # 🖼️ Load image and resize
        img = load_img(image_path, target_size=IMG_SIZE)

        # 🔢 Convert image to array & normalize (0-1)
        img_array = img_to_array(img) / 255.0

        # ➕ Add batch dimension (required for model)
        img_array = np.expand_dims(img_array, axis=0)

        # 🤖 Predict using model
        preds = model.predict(img_array)

        # 🧠 Get predicted class index
        class_index = int(np.argmax(preds, axis=1)[0])

        # 📊 Get confidence %
        confidence = float(np.max(preds) * 100)

        # 🏷️ Get label name
        tumor_label = CLASS_LABELS[class_index]

        # 🔁 Convert label to user-friendly text
        prediction = "NO TUMOR" if tumor_label == "notumor" else tumor_label.upper()


    # 💾 Store data in session (used for PDF)
    session['patient_name'] = patient_name
    session['patient_id'] = patient_id
    session['age'] = age
    session['doctor'] = doctor
    session['prediction'] = prediction
    session['confidence'] = round(confidence, 2)
    session['image_path'] = image_path


    # 📄 Show result page
    return render_template(
        "result.html",
        patient_name=patient_name,
        patient_id=patient_id,
        age=age,
        doctor=doctor,
        prediction=prediction,
        confidence=round(confidence, 2),
    )


# 📄 PDF Download Route
@app.route("/download_pdf")
def download_pdf():

    # 📥 Get stored data from session
    patient_name = session.get('patient_name', 'Unknown')
    patient_id   = session.get('patient_id', 'N/A')
    age          = session.get('age', 'N/A')
    doctor       = session.get('doctor', 'N/A')
    prediction   = session.get('prediction', 'N/A')
    confidence   = session.get('confidence', 2)
    image_path   = session.get('image_path', None)

    # 📁 File name for PDF
    filename = f"Report_{patient_id}.pdf"

    # 📄 Create PDF canvas
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # 🏷️ Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 80, "MRI Brain Tumor Detection Report")

    # 📅 Date & Time
    now_str = datetime.now().strftime("%d-%m-%Y %H:%M")
    c.setFont("Helvetica", 10)
    c.drawString(100, height - 100, f"Generated on: {now_str}")

    # 👤 Patient Details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 140, "Patient Details")

    c.setFont("Helvetica", 12)
    c.drawString(120, height - 160, f"Name      : {patient_name}")
    c.drawString(120, height - 180, f"Patient ID: {patient_id}")
    c.drawString(120, height - 200, f"Age       : {age}")
    c.drawString(120, height - 220, f"Doctor    : {doctor}")

    # 🤖 Prediction Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 260, "Prediction")

    c.setFont("Helvetica", 12)
    c.drawString(120, height - 280, f"Tumor Type : {prediction}")
    c.drawString(120, height - 300, f"Confidence : {confidence}%")

    # 🖼️ Add MRI image (if exists)
    if image_path and os.path.exists(image_path):
        try:
            img = ImageReader(image_path)
            c.drawImage(img, 100, height - 550, width=200, height=200)
        except:
            pass

    # ✅ Save PDF
    c.showPage()
    c.save()

    # 📤 Send file to user
    return send_file(filename, as_attachment=True)


# ▶️ Run Flask server
if __name__ == "__main__":
    app.run(debug=True)