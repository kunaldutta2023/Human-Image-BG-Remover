from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model

# =====================================
# FLASK APP
# =====================================

app = Flask(__name__)

# =====================================
# CONFIGURATION
# =====================================

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
MODEL_PATH = "models/best_model.keras"

IMG_SIZE = 128

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

# Create folders automatically

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================

print("Loading AI Model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model Loaded Successfully!")

# =====================================
# HELPER FUNCTIONS
# =====================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# =====================================
# ROUTES
# =====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =====================================
# UPLOAD IMAGE
# =====================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    try:

        # -------------------------
        # Validate File
        # -------------------------

        if "image" not in request.files:

            return render_template(
                "index.html",
                error="No image selected."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select an image."
            )

        if not allowed_file(file.filename):

            return render_template(
                "index.html",
                error="Only JPG, JPEG and PNG files are allowed."
            )

        # -------------------------
        # Save Upload
        # -------------------------

        filename = secure_filename(
            file.filename
        )

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(upload_path)

        # -------------------------
        # Read Image
        # -------------------------

        img = cv2.imread(
            upload_path
        )

        if img is None:

            return render_template(
                "index.html",
                error="Unable to read image."
            )

        original = img.copy()

        h, w = original.shape[:2]

        # -------------------------
        # Preprocess
        # -------------------------

        image = cv2.resize(
            img,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        # -------------------------
        # Predict Mask
        # -------------------------

        prediction = model.predict(
            image,
            verbose=0
        )[0]

        print(
            "Prediction Shape:",
            prediction.shape
        )

        # -------------------------
        # Convert Prediction
        # -------------------------

        mask = (
            prediction > 0.5
        ).astype(np.uint8)

        # Handle different output shapes

        if len(mask.shape) == 3:

            mask = mask[:, :, 0]

        mask = cv2.resize(
            mask,
            (w, h)
        )

        print(
            "Mask Shape:",
            mask.shape
        )

        # -------------------------
        # Create Transparent PNG
        # -------------------------

        rgba = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2BGRA
        )

        rgba[:, :, 3] = mask * 255

        # -------------------------
        # Save Output
        # -------------------------

        output_name = (
            filename.rsplit(".", 1)[0]
            + "_removed.png"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        cv2.imwrite(
            output_path,
            rgba
        )

        # -------------------------
        # Show Result Page
        # -------------------------

        return render_template(
            "result.html",
            original_image=filename,
            output_image=output_name
        )

    except Exception as e:

        print("ERROR:", str(e))

        return f"""
        <h1>Error Occurred</h1>
        <p>{str(e)}</p>
        """

# =====================================
# ERROR PAGES
# =====================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "index.html"
    ), 404

@app.errorhandler(500)
def server_error(error):

    return f"""
    <h1>Server Error</h1>
    <p>{error}</p>
    """, 500

# =====================================
# RUN APP
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )