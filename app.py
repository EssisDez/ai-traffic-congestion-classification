
import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "traffic_congestion_mobilenetv2.keras"
)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

def predict_traffic(image):
    if image is None:
        return {}

    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32)

    # Keep preprocessing consistent with the model's training pipeline.
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array, verbose=0)[0]

    return {
        str(class_names[i]).capitalize(): float(predictions[i])
        for i in range(len(class_names))
    }

demo = gr.Interface(
    fn=predict_traffic,
    inputs=gr.Image(type="pil", label="Upload a traffic or road image"),
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted congestion level"
    ),
    title="AI-Powered Traffic Congestion Classification",
    description=(
        "A CNN transfer-learning project using MobileNetV2 "
        "to classify traffic images as Low, Moderate, or High congestion."
    )
)

demo.launch(share=True, server_name="0.0.0.0")
