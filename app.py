from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the Keras models
model_1 = load_model('models/CNN_model_RTSR.h5')  # Change to your model filenames
model_2 = load_model('models/alexnet_model_RTSR.h5')
model_3 = load_model('models/vgg19_model_RTSR.h5')

# Traffic sign classes (as an example, you can modify these)
classes = {
    1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)', 4: 'Speed limit (60km/h)', 
    5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)', 7: 'End of speed limit (80km/h)', 8: 'Speed limit (100km/h)', 
    9: 'Speed limit (120km/h)', 10: 'No passing', 11: 'No passing vehicles over 3.5 tons', 12: 'Right of way at intersection', 
    13: 'Priority road', 14: 'Yield', 15: 'Stop', 16: 'No vehicles', 17: 'Vehicles > 3.5 tons prohibited', 18: 'No entry', 
    19: 'General caution', 20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve', 23: 'Bumpy road', 
    24: 'Slippery road', 25: 'Road narrows on the right', 26: 'Road work', 27: 'Traffic signals', 28: 'Pedestrians', 
    29: 'Children crossing', 30: 'Bicycles crossing', 31: 'Beware of ice/snow', 32: 'Wild animals crossing', 
    33: 'End speed and passing limits', 34: 'Turn right ahead', 35: 'Turn left ahead', 36: 'Ahead only', 
    37: 'Go straight or right', 38: 'Go straight or left', 39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory', 
    42: 'End of no passing', 43: 'End no passing for vehicles > 3.5 tons'
}

# Define the path for saving uploaded files
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image_for_model_1(image):
    # Add specific preprocessing steps for model 1
    image = image.resize((30, 30))
    # Convert image to numpy array
    image = np.array(image)
    # Normalize image data (scaling pixel values to [0, 1])
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_image_for_model_2(image):
    # Add specific preprocessing steps for model 2
    image = image.resize((227, 227))
    image = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_image_for_model_3(image,out_size):
    # Add specific preprocessing steps for model 3
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
    height, width = image.shape[:2]
    scale = out_size / max(height, width)
    new_size = (int(scale * width), int(scale * height))
    image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # Create a new blank image and place the resized image in the center
    result_image = np.zeros((out_size, out_size, 3), dtype=np.float32)  # Black background
    y_offset = (out_size - new_size[1]) // 2
    x_offset = (out_size - new_size[0]) // 2
    result_image[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = image_resized / 255.0  # Normalize
    
    return np.expand_dims(result_image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the image
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Predict with each model
        processed_image_1 = preprocess_image_for_model_1(image)
        pred_1 = model_1.predict(processed_image_1)
        sign_1 = classes.get(np.argmax(pred_1[0]) + 1, "Unknown")

        processed_image_2 = preprocess_image_for_model_2(image)
        pred_2 = model_2.predict(processed_image_2)
        sign_2 = classes.get(np.argmax(pred_2[0]) + 1, "Unknown")

        processed_image_3 = preprocess_image_for_model_3(image,50)
        pred_3 = model_3.predict(processed_image_3)
        sign_3 = classes.get(np.argmax(pred_3[0]) + 1, "Unknown")

        return render_template('index.html', prediction_1=sign_1, prediction_2=sign_2, prediction_3=sign_3, img_path=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

