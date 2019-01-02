from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import create_features
import pandas as pd

class logistic_classifier(object):
    def __init__(self):
        self.scaler = None
        self.logreg = None

    def fit(self, X_train: pd.DataFrame, y_train):
        X_train_log, new_columns = create_features.create_new_features(X_train)
        self.scaler = StandardScaler()
        X_train_log = self.scaler.fit_transform(X_train_log)
        self.logreg = LogisticRegression()
        self.logreg.fit(X_train_log, y_train)

    def prediction_and_confidence(self, X: pd.DataFrame):
        X_test_log, new_columns = create_features.create_new_features(X)
        X_test_log = self.scaler.transform(X_test_log)
        test_prediction = self.logreg.predict(X_test_log)
        test_confidence = self.logreg.decision_function(X_test_log)
        return test_prediction,test_confidence
