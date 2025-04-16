from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import keras

app = Flask(__name__)

# Load your model once
model = keras.models.load_model('EfficientNetB0_model.h5')

# List of class names
classes = [
    "Normal", "Diabetic Retinopathy", "Glaucoma", "Myopia",
    "AMD", "Hypertension", "Not an eye Image", "Others"
]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']  # Get image from Flutter app
    img = Image.open(file.stream).resize((224, 224))
    img_array = np.array(img.convert('RGB'), dtype=np.float64).reshape(1, 224, 224, 3)
    
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return jsonify({
        'disease': classes[predicted_class]
    })

if __name__ == '__main__':
    app.run(debug=False)
