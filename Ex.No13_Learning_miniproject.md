# Ex.No: 13 Learning – Medical Diagnosis System
### DATE: 13/05/2025                                                                        
### REGISTER NUMBER : 212222040182
---
### AIM: 
To write a program to train the classifier for Linear Regression, Support Vector Model, Logistic Regression, Random Forest, etc.

###  Algorithm:

1. *Import Libraries*  
   - Import necessary libraries such as pandas, numpy, and sklearn.

2. *Load Dataset*  
   - Read the dataset (e.g., heart disease, diabetes) using pandas.

3. *Preprocess Data*  
   - Handle missing values.  
   - Encode categorical variables.  
   - Normalize/scale features.  

4. *Split Data*  
   - Divide the dataset into training and testing sets using train_test_split.  

5. *Train Model*  
   - Use a supervised learning algorithm (e.g., Random Forest, Logistic Regression, SVM).  

6. *Evaluate Model*  
   - Predict on test data.  
   - Evaluate performance using metrics like accuracy, precision, recall, and F1-score.  

7. *Save Model*  
   - Export the trained model using joblib or pickle for future use.  

### Program: Heart Disease Prediction

python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("heart.csv")  # Update with your dataset name

# Split into features and label
X = data.drop("target", axis=1)
y = data["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Heart Disease Prediction Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "heart_disease_model.sav")



### Program: 2. Diabetes Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Diabetes Accuracy:", accuracy)

# Save model
joblib.dump(model, "diabetes_model.sav")



### Program: 3. Thyroid Disease Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("thyroid.csv")

# Assuming 'target' is the label column
X = data.drop("target", axis=1)
y = data["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Thyroid Prediction Accuracy:", accuracy)

# Save model
joblib.dump(model, "thyroid_model.sav")


### Program: 4. Lung Cancer Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("lung_cancer.csv")

# Assume 'Level' is the target
X = data.drop("Level", axis=1)
y = data["Level"]

# Encode target if necessary
y = y.astype('category').cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Lung Cancer Prediction Accuracy:", accuracy)

# Save model
joblib.dump(model, "lungs_disease_model.sav")



### Program: 5. Parkinson’s Disease Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("parkinsons.csv")

# Drop non-numeric or irrelevant columns
X = data.drop(["name", "status"], axis=1)
y = data["status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Parkinson’s Accuracy:", accuracy)

# Save model
joblib.dump(model, "parkinsons_model.sav")




## Output:
![WhatsApp Image 2025-05-13 at 11 42 23_069ddd05](https://github.com/user-attachments/assets/b5cd0c06-9b51-45cb-967e-1b1d7261fab5)




### Result:
Thus the system was trained successfully and the prediction was carried out.
