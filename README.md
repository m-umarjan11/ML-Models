# Decision Tree Classification with Post-Pruning

This repository contains code for implementing decision tree classification with post-pruning. Decision trees are powerful machine learning models used for both classification and regression tasks. However, they are prone to overfitting, especially when the tree depth is not limited. Post-pruning is a technique used to mitigate overfitting by removing nodes from the tree that do not improve its predictive accuracy on a validation set.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/decision-tree-classification.git
    ```

2. Navigate to the project directory:

    ```
    cd decision-tree-classification
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

To use the decision tree classifier with post-pruning, follow these steps:

1. Import the `DecisionTreeClassifier` class from the `decision_tree.py` module.
2. Create an instance of the `DecisionTreeClassifier` class, specifying parameters such as maximum depth and minimum samples per leaf.
3. Fit the classifier to your training data using the `fit()` method.
4. Evaluate the performance of the classifier using appropriate metrics such as accuracy, precision, recall, etc.
5. Optionally, visualize the decision tree using tools like Graphviz.

## Example

Here's an example of how to use the decision tree classifier with post-pruning in Python:

```python
from decision_tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create and fit the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
