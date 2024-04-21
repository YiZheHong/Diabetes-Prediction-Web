# Diabetes Prediction Web Development 

This project aims to develop a robust machine learning model that can predict diabetes from clinical datasets. 
I utilize machine learning model to predict diabetes and build a website for free access, aiming to enhance understanding of medical examinations, benefit from early detection of potential health issues, and cope with the global shortage of medical resources.

## Table of Contents

- [Installation](#installation)
- [Project_Explained](#project_explained)

## Installation

**Step 1: Clone the Repository**

To get started with this project, clone this repository onto your local machine by using the following command:

```bash
git clone https://github.com/YiZheHong/Diabetes-Prediction-Web.git
cd Diabetes-Prediction-Web
```

**Step 2: Set up a Python Environment**

Set up a virtual environment and activate it (Windows):

```bash
python -m venv venv
.\venv\Scripts\activate
```

Or on macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install Requirements**

Install all the required packages using the following command:

```bash
pip install -r requirements.txt
```

This command will install all the necessary Python libraries including `pandas`, `sklearn`, and others which are required for the project.

## Project_Explained

### Data Preprocessing

I began by loading the diabetes dataset and proceeded with initial data exploration. Understanding the shape and basic statistics of the dataset helped me in strategizing the preprocessing steps:

- **Separating Features and Target**: Distinguished between the dataset features (`X`) and the target variable (`y`).

- **Handling Categorical and Numerical Data**: Utilized pipelines to apply appropriate transformations:
  - **Numerical Data**: Imputation of missing values with the mean and scaling the data.
  - **Categorical Data**: Imputation with the most frequent category followed by one-hot encoding.

### Class Imbalance Handling

To ensure the model learns equally from both classes, I addressed the class imbalance by downsampling the majority class. This was crucial for preventing the model from being biased towards the majority class.

### Feature Selection

Using `SelectKBest` with `f_classif`, I selected the top 4 features that showed the most significant relationships with the target variable. This not only improved the model's performance but also reduced the computational cost.

### Model Training

The preprocessed and feature-selected data was then split into training and testing sets. A stratified split was used to maintain the proportion of the target class. The training set was used to train the model, and the testing set was used to evaluate its performance.

### Model Selection

#### Overview

For the diabetes prediction task, I explored several machine learning models to determine the most effective approach for our dataset. The primary models considered were the Multi-Layer Perceptron (MLP), Random Forest, and Support Vector Machine (SVM). Each of these models brings unique strengths to handling classification tasks and was evaluated based on their performance, ease of interpretation, computational efficiency, and suitability to the data at hand.

#### Models Considered

**1. Multi-Layer Perceptron (MLP)**:
   - **Description**: MLP is a type of neural network that consists of at least three layers of nodes: an input layer, hidden layers, and an output layer. The nodes are interconnected through weights and use a nonlinear activation function, typically ReLU.
   - **Strengths**: Capable of capturing complex nonlinear relationships in the data through its multiple layers and nonlinearity.
   - **Weaknesses**: Can be quite sensitive to feature scaling and often requires tuning of parameters such as the number of hidden neurons, layers, and regularization methods.
   - **Rationale for use**: Given the complex nature of medical data and the potential nonlinear relationships between features, MLP was considered to capture these complexities effectively.

**2. Random Forest**:
   - **Description**: This ensemble learning method operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees.
   - **Strengths**: Offers high accuracy, handles bias well, and provides feature importance — useful for interpretation and understanding of the model.
   - **Weaknesses**: Although it generally performs well, it can overfit if not tuned properly and is computationally intensive, making it less efficient on larger datasets.
   - **Rationale for use**: Random Forest was chosen for its robustness to outliers and its effectiveness in classification tasks, particularly because it handles imbalanced data well, which is a noted challenge in this project.

**3. Support Vector Machine (SVM)**:
   - **Description**: SVM constructs a hyperplane in a high-dimensional space to separate different classes. SVM aims at maximizing the margin between the closest members of separate classes.
   - **Strengths**: Effective in high-dimensional spaces and relatively memory efficient.
   - **Weaknesses**: Does not perform well with large datasets and can be tricky to tune due to the importance of picking the right kernel.
   - **Rationale for use**: SVM was considered because of its effectiveness in binary classification problems, especially with a clear margin of separation and when the number of dimensions exceeds the number of samples.

#### Final Choice: Multi-Layer Perceptron (MLP)

**Why MLP was Chosen**:
- **Capability to Model Non-Linearities**: The ability of MLP to learn non-linear functions and capture complex relationships in the data makes it particularly suitable for medical datasets, where many features can interact in non-linear ways to affect the outcome.
- **Scalability and Adaptability**: Once correctly configured, MLPs are highly adaptable to new problems simply by retraining or adding layers/neurons, making them very versatile for varying types of data inputs and complexities.
- **Performance**: In trials, MLP provided the best compromise between accuracy and computational efficiency on the validation set. It managed to achieve high accuracy while correctly balancing the precision and recall, which are crucial metrics for medical predictions.
- **Handling of Feature Interactions**: MLP's structure allows it to implicitly detect all possible interactions between features, unlike models like SVM or Random Forest, which might require explicit feature engineering to capture these interactions.

