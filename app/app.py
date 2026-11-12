
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf
import joblib
import sqlite3
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

from PIL import Image
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, session,
    send_from_directory, url_for
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_FILE = BASE_DIR / "database" / "neuroscan.db"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static")
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.secret_key = os.environ.get(
    "NEUROSCAN_SECRET_KEY",
    "development-only-change-this-before-deployment"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}

# Load the trained models once
feature_extractor = tf.keras.models.load_model(
    MODEL_DIR / "densenet121_feature_extractor.keras",
    compile=False
)

svm_model = joblib.load(MODEL_DIR / "densenet121_svm.pkl")
scaler = joblib.load(MODEL_DIR / "densenet121_svm_scaler.pkl")


def get_db():
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection



def initialize_database():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                mobile TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT,
                gradcam_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER
            )
        """)

        columns = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(prediction_history)"
            ).fetchall()
        ]

        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE prediction_history ADD COLUMN user_id INTEGER"
            )


initialize_database()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


CLASS_NAMES = {
    0: "Glioma",
    1: "Healthy",
    2: "Meningioma",
    3: "Pituitary"
}


def predict_mri(image_path):
    # Read MRI as grayscale
    gray_image = Image.open(image_path).convert("L")

    # Resize and create three equal grayscale channels
    resized = gray_image.resize((224, 224))
    rgb_compatible = resized.convert("RGB")

    image_array = np.asarray(
        rgb_compatible, dtype=np.float32
    )
    image_array = np.expand_dims(image_array, axis=0)

    # DenseNet preprocessing
    processed = tf.keras.applications.densenet.preprocess_input(
        image_array.copy()
    )

    # Feature extraction and SVM prediction
    features = feature_extractor.predict(processed, verbose=0)
    scaled_features = scaler.transform(features)

    predicted_id = int(svm_model.predict(scaled_features)[0])
    probabilities = svm_model.predict_proba(scaled_features)[0]

    confidence = float(
        probabilities[list(svm_model.classes_).index(predicted_id)] * 100
    )

    return CLASS_NAMES[predicted_id], confidence




def generate_occlusion_map(image_path, predicted_label, output_path):
    """
    Generate an Occlusion Sensitivity explanation for the
    DenseNet121 + SVM predicted class. This is NOT standard Grad-CAM.
    """
    # Match the model's existing grayscale-to-RGB preprocessing
    gray = Image.open(image_path).convert("L").resize((224, 224))
    original = np.asarray(gray, dtype=np.float32)

    rgb = np.asarray(gray.convert("RGB"), dtype=np.float32)
    batch = tf.keras.applications.densenet.preprocess_input(
        np.expand_dims(rgb.copy(), axis=0)
    )

    # Baseline probability for the predicted class
    baseline_features = feature_extractor.predict(batch, verbose=0)
    baseline_scaled = scaler.transform(baseline_features)
    baseline_probs = svm_model.predict_proba(baseline_scaled)[0]

    class_ids = list(svm_model.classes_)
    target_id = next(
        key for key, value in CLASS_NAMES.items()
        if value == predicted_label
    )
    target_index = class_ids.index(target_id)
    baseline_score = float(baseline_probs[target_index])

    # Occlude 48x48 patches with the image's mean intensity
    patch_size = 48
    stride = 24
    fill_value = float(np.mean(original))

    positions = [
        (y, x)
        for y in range(0, 224 - patch_size + 1, stride)
        for x in range(0, 224 - patch_size + 1, stride)
    ]

    occluded_images = []
    for y, x in positions:
        modified = original.copy()
        modified[y:y+patch_size, x:x+patch_size] = fill_value

        rgb_modified = np.repeat(
            modified[..., np.newaxis], 3, axis=-1
        )
        processed = tf.keras.applications.densenet.preprocess_input(
            rgb_modified.astype(np.float32)
        )
        occluded_images.append(processed)

    occluded_batch = np.stack(occluded_images, axis=0)

    features = feature_extractor.predict(
        occluded_batch, batch_size=16, verbose=0
    )
    scaled = scaler.transform(features)
    probabilities = svm_model.predict_proba(scaled)

    # Probability decrease indicates sensitivity to the occluded region
    heatmap = np.zeros((224, 224), dtype=np.float32)
    counts = np.zeros((224, 224), dtype=np.float32)

    for (y, x), probs in zip(positions, probabilities):
        drop = max(0.0, baseline_score - float(probs[target_index]))
        heatmap[y:y+patch_size, x:x+patch_size] += drop
        counts[y:y+patch_size, x:x+patch_size] += 1.0

    heatmap /= np.maximum(counts, 1.0)

    maximum = float(np.max(heatmap))
    if maximum > 1e-8:
        heatmap /= maximum
    else:
        # A flat map is more honest than inventing an important region
        heatmap[:] = 0.0

    # Create a red overlay on the grayscale MRI
    intensity = np.clip(heatmap, 0.0, 1.0)
    base_rgb = np.asarray(gray.convert("RGB"), dtype=np.float32)

    red_overlay = np.zeros_like(base_rgb)
    red_overlay[..., 0] = 255.0

    alpha = (intensity * 0.55)[..., np.newaxis]
    blended = base_rgb * (1.0 - alpha) + red_overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    Image.fromarray(blended).save(output_path, format="PNG")
    return output_path


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "")

        with get_db() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE mobile = ?",
                (mobile,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["full_name"] = user["full_name"]
            session["mobile"] = user["mobile"]
            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            auth_message="Invalid mobile number or password."
        ), 401

    return render_template("login.html", auth_message="")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not mobile or not password:
            return render_template(
                "register.html",
                auth_message="Please complete all fields."
            ), 400

        if len(password) < 8:
            return render_template(
                "register.html",
                auth_message="Password must contain at least 8 characters."
            ), 400

        if password != confirm_password:
            return render_template(
                "register.html",
                auth_message="Passwords do not match."
            ), 400

        try:
            with get_db() as connection:
                connection.execute(
                    """
                    INSERT INTO users (full_name, mobile, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (
                        full_name,
                        mobile,
                        generate_password_hash(password)
                    )
                )

            return render_template(
                "login.html",
                auth_message="Account created successfully. Please sign in."
            )

        except sqlite3.IntegrityError:
            return render_template(
                "register.html",
                auth_message="This mobile number is already registered."
            ), 409

    return render_template("register.html", auth_message="")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM prediction_history WHERE user_id = ?",
            (session["user_id"],)
        ).fetchone()[0]

    return render_template(
        "dashboard.html",
        user_name=session.get("full_name", "User"),
        total_predictions=total
    )


@app.route("/history")
@login_required
def history():
    with get_db() as connection:
        records = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (session["user_id"],)
        ).fetchall()

    return render_template(
        "history.html",
        history_records=records
    )



@app.route("/gradcam-demo")
@login_required
def gradcam_demo():
    return render_template("gradcam_demo.html")

@app.route("/health")
def health():
    return {"status": "OK", "message": "NeuroScan AI is running"}


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/result")
def result():
    return render_template(
        "result.html",
        prediction=None,
        confidence="--",
        image_url=None,
        gradcam_url=None
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html", upload_message="")

    uploaded_file = request.files.get("mriFile")

    if uploaded_file is None or not uploaded_file.filename:
        return render_template(
            "upload.html",
            upload_message="Please select an MRI image."
        ), 400

    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        return render_template(
            "upload.html",
            upload_message="Only JPG, JPEG and PNG files are allowed."
        ), 400

    try:
        # Verify the uploaded image
        check_image = Image.open(uploaded_file.stream)
        actual_format = check_image.format
        check_image.verify()

        expected_formats = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG"
        }

        if actual_format != expected_formats[extension]:
            raise ValueError("File extension does not match image format.")

        uploaded_file.stream.seek(0)
        original_image = Image.open(uploaded_file.stream)

        if original_image.width < 1 or original_image.height < 1:
            raise ValueError("Invalid image dimensions.")

        # Save original upload without changing its image content
        unique_id = uuid4().hex[:10]
        original_stem = Path(original_name).stem
        saved_name = f"{original_stem}_{unique_id}{extension}"
        original_path = UPLOAD_DIR / saved_name

        uploaded_file.stream.seek(0)
        uploaded_file.save(original_path)

        # Create a separate grayscale preview
        gray_name = f"{original_stem}_{unique_id}_gray.png"
        gray_path = UPLOAD_DIR / gray_name

        with Image.open(original_path) as source:
            gray_preview = source.convert("L")
            gray_preview.save(gray_path, format="PNG")

        # Run the original trained DenseNet121 + SVM model
        prediction, confidence = predict_mri(original_path)

        # Temporarily skip XAI to test prediction speed
        xai_url = None
        saved_xai_name = None

        connection = sqlite3.connect(DB_FILE)
        connection.execute(
            """
            INSERT INTO prediction_history
            (filename, prediction, confidence, image_path, gradcam_path, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                saved_name, prediction, confidence, gray_name,
                saved_xai_name, session["user_id"]
            )
        )
        connection.commit()
        connection.close()

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=f"{confidence:.2f}%",
            image_url=url_for("uploaded_file", filename=gray_name),
            gradcam_url=xai_url,
            xai_label="XAI temporarily disabled for testing"
        )

    except Exception as error:
        print("MRI processing error:", repr(error))
        return render_template(
            "upload.html",
            upload_message=(
                "MRI processing failed. Please upload a valid JPG, JPEG or PNG image."
            )
        ), 400


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "upload.html",
        upload_message="File is too large. Maximum size is 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=False)
