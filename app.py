from flask import Flask, render_template, request
from joblib import load
nn_clf = load('Diabetes_MLP.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input data
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        blood_glucose = float(request.form['blood_glucose'])
        hypertension = float(request.form['hypertension'])

        # Create a sample data point
        user_data = [[age, hypertension, bmi, hba1c, blood_glucose]]

        # Make a prediction
        prediction = nn_clf.predict(user_data)[0]
        if prediction == 0:
            result = "You do not have diabetes."
        else:
            result = "You have diabetes."

        return render_template('index1.html', result=result)

    return render_template('index1.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)