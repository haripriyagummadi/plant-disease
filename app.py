from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np 
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')
print("Model loaded successfully!")

# Define labels and treatment advice
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
disease_treatments = {
    'Rust': [
        "Remove affected leaves: Pick off affected leaves as soon as you see them.",
        "Dispose of infected plants: Remove and destroy severely affected plants to prevent the spread of spores.",
        "Use fungicides: Apply fungicides as soon as you see signs of rust.",
        "You can use products from different modes of action groups.",
        "Some fungicides that are effective against rust include mancozeb, myclobutanil, and triadimefon.",
        "Avoid overhead irrigation: Avoid overhead irrigation, especially in the evening, to reduce wet foliage.",
        "Improve air circulation: Improve air circulation by pruning overhanging trees and shrubs.",
        "Control weeds: Control weeds that can act as alternate hosts for rusts.",
        "Quarantine new plants: Keep new plants of susceptible species in quarantine for a week.",
        "Treat roses: For rose leaf rust, prune out infected stems in the spring.",
        "If the rose also has aphids, you can spray with a combined insecticide and fungicide.",
        "Treat pear trees: Remove any juniper bushes in your garden, as they can host the fungus that causes rust on pear trees.",
        "Avoid applying to flowering plants: Don't apply fungicides to plants when they are in flower, or when bees are actively foraging."
    ],
    'Powdery': [
        "Remove affected areas: Cut off leaves or branches infected with powdery mildew to stop it from spreading.",
        "Apply fungicides: Use fungicides specifically designed for powdery mildew.",
        "Copper fungicides, potassium bicarbonate, and sulfur fungicides are effective.",
        "Use a milk solution: Mix 1 part milk with 2 parts water and spray the affected areas.",
        "The milk compounds help kill mildew in sunlight.",
        "Spray baking soda solution: Mix 1 tablespoon of baking soda with ½ teaspoon of liquid soap in 1 gallon of water.",
        "Spray on the affected areas.",
        "Use neem oil: Neem oil is a natural remedy that can help control powdery mildew infections.",
        "Improve air circulation: Ensure good airflow around the plants by spacing them apart and pruning excess growth.",
        "Avoid overhead watering: Water at the base of the plant to prevent spores from spreading and creating a favorable environment for the mildew.",
        "Don't over-fertilize: Excessive fertilizer can encourage new growth, making plants more susceptible to powdery mildew.",
        "Remove dead or diseased foliage: Regularly remove leaves and plants that show signs of powdery mildew to help maintain overall plant health.",
        "Disinfect gardening tools: Clean pruning shears or other tools after working with infected plants to prevent spreading the disease.",
        "Choose resistant varieties: When planting new crops, select varieties that are resistant to powdery mildew."
    ],
    'Healthy': [
        "Your plant is healthy! Maintain regular care and monitor for any signs of disease."
    ]
}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file!", 400

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
    file.save(file_path)

    predictions = getResult(file_path)
    predicted_label = labels[np.argmax(predictions)]
    treatment_advice = disease_treatments.get(predicted_label, [])

    # Get the filename to display the uploaded image on result page
    file_url = url_for('uploaded_file', filename=secure_filename(file.filename))

    return render_template('result.html', predicted_label=predicted_label, treatment_advice=treatment_advice, file_url=file_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)
