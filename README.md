# Breast Cancer Classification

This project implements a **Breast Cancer Classification** system using machine learning techniques. The goal is to classify tumors as malignant (M) or benign (B) based on various features extracted from breast mass images. The project leverages the **Naive Bayes Algorithm** for classification and includes detailed data preparation, visualization, and model evaluation steps.

---

## Features of the Project

- **Data Preparation**: 
  - Cleaned and preprocessed the dataset to remove unnecessary columns and handle missing values.
  - Split the dataset into training and testing sets with balanced classes.

- **Algorithm Used**: 
  - Implemented the **Naive Bayes Algorithm**, which is based on **Bayes' Theorem**.
  - Assumes features are independent and identically distributed (IID).

- **Interactive Visualizations**:
  - Used libraries like **Seaborn** and **Plotly** to create interactive plots for better understanding of the data.

- **Code Documentation**:
  - Added detailed comments and markdown cells in the code to explain each step.
  - Included tables and images to make the content more engaging and easier to follow.

---

## Dataset Overview

The dataset contains 569 entries with 33 columns. Key features include:

| Feature Name           | Description                          |
|------------------------|--------------------------------------|
| `radius_mean`          | Mean of distances from center to points on the perimeter |
| `texture_mean`         | Standard deviation of gray-scale values |
| `perimeter_mean`       | Mean size of the core tumor         |
| `area_mean`            | Mean area of the tumor              |
| `smoothness_mean`      | Mean of local variation in radius lengths |
| `compactness_mean`     | Mean of perimeter²/area - 1.0       |
| `concavity_mean`       | Mean of severity of concave portions of the contour |
| `concave points_mean`  | Mean for number of concave portions of the contour |
| `symmetry_mean`        | Mean symmetry of the tumor          |
| `fractal_dimension_mean` | Mean "coastline approximation" - 1 |

---

## Steps in the Project

### 1. Data Loading and Cleaning
- The dataset was loaded using `pandas` and unnecessary columns like `id` and `Unnamed: 32` were dropped.
- Verified that there were no null values in the dataset.

```python
# Loading the dataset
raw_data = pd.read_csv("D:\\Document\\Machine_Learning\\Breast_Cancer_Folder\\data.csv")

# Dropping unnecessary columns
raw_data_copy = raw_data.drop(labels=[raw_data.columns[0], raw_data.columns[32]], axis=1)
```

---

### 2. Data Visualization
- Visualized the distribution of features using histograms and scatter plots to understand the data better.
- Example: Distribution of `radius_mean` for malignant and benign tumors.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the distribution of radius_mean for malignant and benign tumors
sns.histplot(data=raw_data_copy, x="radius_mean", hue="diagnosis", kde=True)
plt.title("Distribution of Radius Mean")
plt.show()
```

- Created pair plots to observe relationships between key features.

```python
# Pair plot for selected features
selected_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
sns.pairplot(raw_data_copy, vars=selected_features, hue="diagnosis", diag_kind="kde")
plt.show()
```

---

### 3. Data Preparation
- Encoded the target variable (`diagnosis`) as binary values: `M` (Malignant) → `1`, `B` (Benign) → `0`.
- Split the dataset into training and testing sets using an 80-20 split.

```python
from sklearn.model_selection import train_test_split

# Encoding the target variable
raw_data_copy["diagnosis"] = raw_data_copy["diagnosis"].map({"M": 1, "B": 0})

# Splitting the dataset
X = raw_data_copy.drop("diagnosis", axis=1)
y = raw_data_copy["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 4. Model Training and Evaluation
- Trained the **Naive Bayes Algorithm** on the training data.
- Evaluated the model's performance using metrics like accuracy, precision, recall, and F1-score.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Training the model
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 5. Results
- The model achieved high accuracy in classifying tumors as malignant or benign.
- Example confusion matrix and classification report:

| Metric         | Value  |
|----------------|--------|
| Accuracy       | 95.6%  |
| Precision      | 96.2%  |
| Recall         | 94.8%  |
| F1-Score       | 95.5%  |

---

### 6. Visualizing Model Performance
- Plotted the confusion matrix for better visualization of the model's performance.

```python
import seaborn as sns

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```
---
### Thank you for exploring this project! 😊