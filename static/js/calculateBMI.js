function calculateBMI() {
    var weight = parseFloat(document.getElementById('weight').value);
    var height = parseFloat(document.getElementById('height').value);
    if (height > 0 && weight > 0) {
        var bmi = (weight / (height * height)).toFixed(1); // Calculates BMI and rounds to one decimal place
        var resultText = "";

        if (bmi < 18.5) {
            resultText = "Your BMI is " + bmi + ". This is considered underweight.";
        } else if (bmi >= 18.5 && bmi <= 24.9) {
            resultText = "Your BMI is " + bmi + ". This is considered a normal weight.";
        } else if (bmi >= 25 && bmi <= 29.9) {
            resultText = "Your BMI is " + bmi + ". This is considered overweight.";
        } else {
            resultText = "Your BMI is " + bmi + ". This is considered obese.";
        }

        document.getElementById('bmiResult').innerText = resultText; // Outputs result
    } else {
        document.getElementById('bmiResult').innerText = "Please enter valid weight and height.";
    }
}
