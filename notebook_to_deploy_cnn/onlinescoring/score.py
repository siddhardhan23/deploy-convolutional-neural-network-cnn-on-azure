import os
import json
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize global variables
def init():
    global model
    global class_indices
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'plant_disease_prediction_model.h5')
    model = tf.keras.models.load_model(model_path)

    class_indices_path = 'class_indices.json'
    with open(class_indices_path) as f:
        class_indices = json.load(f)

# Function to load and preprocess the image
def load_and_preprocess_image(image_data, target_size=(224, 224)):
    # Assuming image_data is a base64 encoded string
    decoded_image_data = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(decoded_image_data))
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image
def predict_image_class(image_data):
    preprocessed_img = load_and_preprocess_image(image_data)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Run the prediction
def run(raw_data):
    try:
        # Convert the JSON string to bytes
        image_data = json.loads(raw_data)['data']
        image_data = bytes(image_data, encoding='utf-8')
        predicted_class_name = predict_image_class(image_data)
        return json.dumps({"Predicted Class Name": predicted_class_name})
    except Exception as e:
        error = str(e)
        return error
