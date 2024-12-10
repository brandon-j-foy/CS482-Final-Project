import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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

class MyNN():
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model = MLPClassifier(
            activation='relu', 
            solver='adam', 
            alpha=1e-4, 
            hidden_layer_sizes=(400, 5), 
            random_state=42, 
            learning_rate_init=0.001,
            learning_rate='adaptive',
            max_iter=1, 
            warm_start=True
            )
        
        for epoch in range(500):  # Assuming a maximum of 500 epochs
            self.model.fit(X_train, y_train)

            # Record training loss
            train_loss = self.model.loss_
            self.history['train_loss'].append(train_loss)

            # Record training accuracy
            train_accuracy = self.model.score(X_train, y_train)
            self.history['train_accuracy'].append(train_accuracy)

            if X_test is not None and y_test is not None:
                # Record test loss and accuracy if test data is provided
                y_pred_test = self.model.predict(X_test)
                test_loss = ((y_test - y_pred_test) ** 2).mean()  # Placeholder for test loss calculation
                self.history['test_loss'].append(test_loss)

                test_accuracy = self.model.score(X_test, y_test)
                self.history['test_accuracy'].append(test_accuracy)

            # Early stopping based on convergence
            if epoch > 10 and abs(self.history['train_loss'][-1] - self.history['train_loss'][-2]) < 1e-6:
                break

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
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
    input_file = input("Enter the path to the CSV file for testing and predictions: ")

    try:
        input_data = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading the input file: {e}")
        return

    training_data_path = "loan_data.csv"
    try:
        training_data = pd.read_csv(training_data_path)
    except Exception as e:
        print(f"Error loading the training data: {e}")
        return

    # Preprocess training data
    training_data, label_encoders = preprocess_data(training_data)

    X_train = training_data[["person_age", "person_gender", "person_education", "person_income", 
                             "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
                             "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", 
                             "previous_loan_defaults_on_file"]].values
    y_train = training_data["loan_status"].values  # Target variable

    # Preprocess input data
    try:
        for col in ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]:
            if col in input_data.columns and col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        X_input = input_data[["person_age", "person_gender", "person_education", "person_income", 
                              "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
                              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", 
                              "previous_loan_defaults_on_file"]].values

        y_test = input_data["loan_status"].values  # Actual labels for testing
    except Exception as e:
        print(f"Error processing the input file: {e}")
        return

    # Scale data for the CNN model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_input_scaled = scaler.transform(X_input)

    # CNN Model
    nn_model = MyNN()
    nn_model.fit(X_train_scaled, y_train)
    nn_predictions = nn_model.predict(X_input_scaled)

    # DT Model
    tree_model = MyXGBoostTree()
    tree_model.fit(X_train, y_train)  # No scaling for the tree model
    tree_predictions = tree_model.predict(X_input)

    # LR Model
    lr_model = MyLogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_input)
    print("\nNeural Network Predictions:")
    print(nn_predictions)
    print("\n")
    print("Decision Tree Predictions:")
    print(tree_predictions)
    print("\n")
    print("Logistic Regression Predictions:")
    print(lr_predictions)
    print("\n")
    print("Actual values:")
    print(y_test)
    cnn_accuracy = accuracy_score(y_test, nn_predictions)
    print("\nNeural Network Accuracy on input file:", cnn_accuracy)
    tree_accuracy = accuracy_score(y_test, tree_predictions)
    print("\nDecision Tree Accuracy on input file:", tree_accuracy)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print("\nLogistic Regression Accuracy on input file:", lr_accuracy)
    print("\n")


if __name__ == '__main__':
    main()
