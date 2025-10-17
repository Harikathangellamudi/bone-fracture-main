import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.optimizers import Adam

# optional: disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load models WITHOUT loading old optimizer state
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5", compile=False)
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5", compile=False)
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5", compile=False)
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5", compile=False)

# (Optional) Recompile if you plan to train further, not needed for inference
for m in [model_elbow_frac, model_hand_frac, model_shoulder_frac, model_parts]:
    m.compile(optimizer=Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# categories
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    prediction = np.argmax(chosen_model.predict(images), axis=1)

    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str

# New helper to return scores without breaking existing API
def predict_with_scores(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
        categories = categories_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac
        categories = categories_fracture

    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    probs = chosen_model.predict(images)[0]
    idx = int(np.argmax(probs))
    label = categories[idx]
    return label, probs.tolist()

# High-level analysis returning structured output for UI
def analyze_image(img_path):
    bone_label, bone_probs = predict_with_scores(img_path, 'Parts')
    if bone_label not in categories_parts:
        return {
            "fracture_present": None,
            "bone": None,
            "fracture_type": "Unknown",
            "severity_percent": None
        }

    frac_label, frac_probs = predict_with_scores(img_path, bone_label)
    # fracture probs are ordered as ['fractured', 'normal']
    fractured_prob = float(frac_probs[0]) if len(frac_probs) > 0 else 0.0
    severity_percent = round(fractured_prob * 100, 1)

    if frac_label == 'fractured':
        fracture_type = f"Fracture in {bone_label} (type not classified)"
        fracture_present = True
    else:
        fracture_type = f"No fracture detected in {bone_label}"
        fracture_present = False

    return {
        "fracture_present": fracture_present,
        "bone": bone_label,
        "fracture_type": fracture_type,
        "severity_percent": severity_percent
    }
