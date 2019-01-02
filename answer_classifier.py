from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

import create_features
from bayes_detector import BayesDetector
from logistic_classifier import logistic_classifier


class answer_classifier(object):
    def __init__(self, df):
        self.df = df
        self.ensemble = None

    def print_results(self, headline, y, prediction):
        print("Method: " + headline + ", confusion Matrix:")
        print(confusion_matrix(y, prediction))
        print("Accuracy score   :", accuracy_score(y, prediction))
        print("Precision score  :", precision_score(y, prediction))
        print("Recall score     :", recall_score(y, prediction))
        print("F1 score:", f1_score(y, prediction, average='macro'))

    def bayes_on_text_column(self, X_test, X_train, column, y_test, y_train):
        X_train_bayes_1, X_test_bayes_1 = X_train[column], X_test[column]
        naive_bayes_prediction_train, naive_bayes_prediction_test = self.fit_bayes(X_train_bayes_1, y_train,
                                                                                   X_test_bayes_1,
                                                                                   y_test)
        naive_bayes_prediction_train_int = self.positive_to_boolean(naive_bayes_prediction_train)
        naive_bayes_prediction_test_int = self.positive_to_boolean(naive_bayes_prediction_test)
        self.print_results_train_test("Naive bayes on column [" + column + "] ", y_train, y_test,
                                      naive_bayes_prediction_train_int,
                                      naive_bayes_prediction_test_int)
        return naive_bayes_prediction_train, naive_bayes_prediction_test

    @staticmethod
    def print_statistics(frame):
        print("Columns:", frame.columns)
        print("Head:")
        print(frame.head())
        print("Description:")
        print(frame.describe())
        counts = pd.Series.value_counts(frame['result'])
        print("Result frequency:")
        print(counts)

    def split_df(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        return X_test, X_train, y_test, y_train

    def split_input_output_train_test(self,df):
        y = self.df['result']
        X_test, X_train, y_test, y_train = self.split_df(self.df, y)
        return X_test, X_train, y_test, y_train

    def run_level_1_predictors(self, X, y=None):
        combined_results_train = pd.DataFrame()
        for name, predictor in self.predictors.items():
            train_prediction_int, train_prediction_confidence = predictor.prediction_and_confidence(X)
            combined_results_train[name] = train_prediction_confidence
            if y is not None:
                print("Checking performance on training set for " + name)
                self.print_results(name,y,train_prediction_int)
        return combined_results_train

    def run_train_test(self,df):
        X_test, X_train, y_test, y_train = self.split_input_output_train_test(df)
        self.build_run_classifiers(X_train, y_train)
        prediction_train = self.predict(X_train)
        self.print_results("Ensemble on Train:",y_train, prediction_train)
        prediction_test = self.predict(X_test)
        self.print_results("Ensemble on Test :",y_test, prediction_test)

    def build_run_classifiers(self, X_train,y_train):
        self.print_statistics(self.df)
        print("----------------")
        # Logistic Regression on hand-crafted features

        classifier_object = {
            "Logistic Regression on 2 Keywords" : logistic_classifier()
        }


        self.predictors = dict()
        self.logreg = logistic_classifier()
        self.logreg.fit(X_train, y_train)

        description = "Logistic Regression on 2 Keywords"
        self.predictors[description] = self.logreg

        # Create first-level predictors
        for column in ["title"]:  # , "description", "content"
            description = "Bayesian on column " + column
            bayesian = BayesDetector(column)
            bayesian.fit(X_train, y_train)
            self.predictors[description] = bayesian

        combined_results_train = self.run_level_1_predictors(X_train, y_train)
        self.scaler = StandardScaler()
        combined_results_train = self.scaler.fit_transform(combined_results_train)

        self.ensemble = RandomForestClassifier()
        self.ensemble.fit(combined_results_train, y_train)

    def predict(self, X):
        combined_results = self.run_level_1_predictors(X)
        combined_results = self.scaler.transform(combined_results)
        prediction = self.ensemble.predict(combined_results)
        return prediction

    def classify(self, url, title, description, content):
        data = {"url": url, "title": title, "description": description, "content": content}
        df = pd.DataFrame.from_dict(data)
        result = self.predict(df)