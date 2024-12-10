import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

class MyLogisticRegression():
    def __init__(self):
        self.X = None
        self.y = None
    
    def fit(self, X, y):
        self.X = X
        self.y = y

        self.model = LogisticRegression(max_iter=500)
        self.model.fit(self.X, self.y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X_test, y_test): # Returns the precision, recall, f1_score, and support
        #return self.model.score(X, y)
        y_pred = self.model.predict(X_test)
        #y_pred_binary = (y_pred > 0.5).astype(int) 
        accuracy = accuracy_score(y_test, y_pred)
        # print("The logistic accuracy is", accuracy)
        # precision, recall, f1, support = precision_recall_fscore_support(self.y, y_pred, average=None)
        #print("Logistic:", precision, recall, f1, support)
        # return [accuracy, precision, recall, f1, support]
        return accuracy
    
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
    data, label_encoders = preprocess_data(data)
    scaler = StandardScaler() # The data doesn't converge in a timely manner (500 iterations), so we scale them.
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
    model = MyLogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    #print("Training sample predictions: ", train_pred)
    #print("Testing sample predictions: ", test_pred)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Training accuracy: ", train_score)
    print("Testing accuracy: ", test_score)

    print("Classification Report:")
    print(classification_report(y_test, test_pred))

if __name__ == '__main__':
    main()