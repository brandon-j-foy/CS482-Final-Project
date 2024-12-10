import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

class MyXGBoostTree():
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        # Create a DecisionTreeClassifier with Entropy (Information Gain)
        self.model = XGBClassifier(
            objective='binary:logistic',  # For binary classification
            eval_metric='logloss',       # Evaluation metric
            random_state=42
        )
        self.model.fit(self.X, self.y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):  # This returns accuracy by default
        return self.model.score(X, y)

def preprocess_data(data):
    # Encode categorical features
    label_encoders = {}  # To store encoders for each column
    
    for col in ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save the encoder for possible inverse transformation
    
    return data, label_encoders

def main():
    # Parse data into X and y:
    data = pd.read_csv("C:/Users/guilh/Desktop/ai project/Final_Project/loan_data.csv")
    #data = data.drop(columns=['date', 'file_name'])  # Drop date and file_name columns
    # Preprocess the data
    data, label_encoders = preprocess_data(data)

    X = data[["person_age", "person_gender", "person_education", "person_income", 
              "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"]].values
    
    # Impute missing values in X
    #imputer = SimpleImputer(strategy='mean')  # Impute NaN values with the mean
    #X = imputer.fit_transform(X)
    
    # Target variable
    y = data["loan_status"].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Create the model and fit it
    model = MyXGBoostTree()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_test = accuracy_score(y_test, test_pred)

    print("The training accuracy is: ", accuracy_train)
    print("The testing accuracy is: ", accuracy_test)

    # Detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

if __name__ == '__main__':
    main()


