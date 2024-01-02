from flask import Flask, request, render_template
import pandas as pd
import pickle

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))


def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25.0:
        return "Healthy Weight"
    elif 25.0 <= bmi < 30.0:
        return "Overweight"
    else:
        return "Obesity"


@app.route("/", methods=["GET"])
def home():
    # Render the home page with the form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Extract height and weight from the form
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    # Create a DataFrame from the input for prediction
    input_data = pd.DataFrame({'Height': [height], 'Weight': [weight]})

    # Predict BMI
    predicted_bmi = model.predict(input_data)[0]

    # Convert predicted_bmi to float and round it
    rounded_bmi = round(float(predicted_bmi), 2)

    # Classify BMI
    bmi_classification = classify_bmi(predicted_bmi)

    # Render the index.html with the prediction results
    return render_template("index.html",
                           bmi=rounded_bmi,
                           classification=bmi_classification)


if __name__ == "__main__":
    app.run(debug=True)
