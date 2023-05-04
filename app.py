# Install necessary packages
# !pip install tensorflow pillow numpy requests efficientnet

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import efficientnet.tfkeras as efn
from flask import Flask, request, jsonify

# Define function to generate hashtags for an image
def generate_hashtags(file):
    # Load pre-trained EfficientNet-B7 model
    model = efn.EfficientNetB7(weights='imagenet')

    # Load and preprocess image
    img = image.load_img(file, target_size=(600, 600))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = efn.preprocess_input(x)

    # Extract features from image using EfficientNet-B7 model
    features = model.predict(x)

    # Get top 10 predicted ImageNet classes for the image
    top_classes = tf.keras.applications.imagenet_utils.decode_predictions(features, top=10)[0]

    # Generate hashtags from predicted classes
    hashtags = [f"#{cls[1].replace('_', '')}" for cls in top_classes]

    return hashtags
  
# Initialize Flask app
app = Flask(__name__)

# Define route handler for form submission
@app.route('/generate_hashtags', methods=['POST'])
def process_image():
    # Check if image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    # Save the uploaded file
    image_file = request.files['image']

    # Generate hashtags for the uploaded image
    hashtags = generate_hashtags(image_file)

    # Return the hashtags as a JSON response
    return jsonify(hashtags)

# Define route handler for index page
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Start Flask app
if __name__ == '__main__':
    app.run()