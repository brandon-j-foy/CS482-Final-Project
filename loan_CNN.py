import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
import warnings
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
     # Parse data into X and y:
    data = pd.read_csv("C:/Users/guilh/Desktop/ai project/Final_Project/loan_data.csv")
    # Preprocess the data
    data, label_encoders = preprocess_data(data)

    X = data[["person_age", "person_gender", "person_education", "person_income", 
              "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
              "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"]].values
    
    # Target variable
    y = data["loan_status"].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #Perform feature scalling for better measurements 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Create the model and fit it
    model = MyNN()
    model.fit(X_train_scaled, y_train, X_test=X_test_scaled, y_test=y_test)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_test = accuracy_score(y_test, test_pred)

    print("The training accuracy is: ", accuracy_train)
    print("The testing accuracy is: ", accuracy_test)

    #Print classification records for better understanding of results
    print("Classification Report:")
    print(classification_report(y_test, test_pred))

    #Print confusion matrix for better understanding of results
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    # Initialize cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model using cross-validation
    scores = cross_val_score(model.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    # Output cross-validation results
    print("\nCross-validation scores:", scores)
    print("Mean cross-validation accuracy:", scores.mean())

    # Plot loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(model.history['train_loss'], label='Train Loss')
    plt.plot(model.history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot accuracy over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(model.history['train_accuracy'], label='Train Accuracy')
    plt.plot(model.history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()