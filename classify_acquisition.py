import pickle
import os.path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import create_features
from bayes_detector import BayesDetector


def print_statistics(frame):
    print("Columns:", frame.columns)
    print("Head:")
    print(frame.head())
    print("Description:")
    print(frame.describe())
    counts = pd.Series.value_counts(frame['result'])
    print("Result frequency:")
    print(counts)


def read_data_from_csv(filename):
    frame = pd.read_csv(filename)
    frame = frame.dropna(axis=0, subset=['class'])
    frame = frame.astype({'class': int, 'url': str, 'title': str, 'description': str, 'content': str})
    frame.fillna(value={'url': '', 'title': '', 'description': '', 'content': ''})
    frame.rename(index=str, columns={'Unnamed: 0': 'number', 'class': 'result'}, inplace=True)
    frame.set_index('number', inplace=True)
    return frame


def read_data():
    csv_filename = "gold_data.csv"
    pickle_filename = "data.pkl"

    if not os.path.isfile(pickle_filename):
        df = read_data_from_csv(csv_filename)
        pickle.dump(df, open(pickle_filename, "wb"))
    else:
        df = pickle.load(open(pickle_filename, "rb"))
    return df


# General initialization
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


def fit_logistic(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    test_prediction = logreg.predict(X_test)
    train_prediction = logreg.predict(X_train)
    test_confidence = logreg.decision_function(X_test)
    train_confidence = logreg.decision_function(X_train)
    # print('Accuracy of Logistic regression classifier on training set: {:.2f}'
    #       .format(logreg.score(X_train, y_train)))
    # print('Accuracy of Logistic regression classifier on test set: {:.2f}'
    #       .format(logreg.score(X_test, y_test)))
    return train_prediction, test_prediction, train_confidence, test_confidence


def fit_bayes(X_train, y_train, X_test, y_test):
    det = BayesDetector()

    det.fit(X_train, y_train)
    train_prediction = det.predict(X_train)
    test_prediction = det.predict(X_test)

    return train_prediction, test_prediction


def print_results(y_test, prediction):
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))
    print("Accuracy score   :", accuracy_score(y_test, prediction))
    print("Precision score  :", precision_score(y_test, prediction))
    print("Recall score     :", recall_score(y_test, prediction))
    print("F1 score:", f1_score(y_test, prediction, average='macro'))


def split_df(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_test, X_train, y_test, y_train


def scale_features(X_test, X_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_test, X_train


def positive_to_boolean(result: list):
    res = pd.Series(result)
    res = res.where(res < 0, 1)
    res = res.where(res >= 0, 0)
    return res


def print_results_train_test(method, y_train, y_test, predict_train, predict_test):
    print("\r\n\r\n" + method + " regression results on train")
    print_results(y_train, predict_train)
    print("\r\n\r\n" + method + " regression results on test")
    print_results(y_test, predict_test)
    print("--------------------")


def bayes_on_text_column(X_test, X_train, column, y_test, y_train):
    X_train_bayes_1, X_test_bayes_1 = X_train[column], X_test[column]
    naive_bayes_prediction_train, naive_bayes_prediction_test = fit_bayes(X_train_bayes_1, y_train, X_test_bayes_1,
                                                                          y_test)
    naive_bayes_prediction_train_int = positive_to_boolean(naive_bayes_prediction_train)
    naive_bayes_prediction_test_int = positive_to_boolean(naive_bayes_prediction_test)
    print_results_train_test("Naive bayes on column [" + column + "] ", y_train, y_test,
                             naive_bayes_prediction_train_int,
                             naive_bayes_prediction_test_int)
    return naive_bayes_prediction_train, naive_bayes_prediction_test


def main():
    df = read_data()
    df, new_columns = create_features.create_new_features(df)
    print("New columns:\r\n", new_columns)
    print_statistics(df)
    print("----------------")

    y = df['result']
    X_test, X_train, y_test, y_train = split_df(df, y)

    # Logistic Regression on hand-crafted features
    X_test_log, X_train_log = X_test[new_columns], X_train[new_columns]
    X_test_log, X_train_log = scale_features(X_test_log, X_train_log)
    logreg_train_prediction, logreg_test_prediction, logreg_train_confidence, logreg_test_confidence = fit_logistic(
        X_train_log, y_train, X_test_log, y_test)

    combined_results_train = pd.DataFrame()
    combined_results_test = pd.DataFrame()
    combined_results_train['logreg'] = logreg_train_confidence
    combined_results_test['logreg'] = logreg_test_confidence

    print("Logistic regression results:")
    print_results(y_test, logreg_test_prediction)
    print_results_train_test("Logistic regression", y_train, y_test, logreg_train_prediction, logreg_test_prediction)

    for column in ["title", "description", "content"]:  #
        naive_bayes_prediction_train, naive_bayes_prediction_test = bayes_on_text_column(X_test, X_train, column,
                                                                                         y_test, y_train)
        combined_results_train["bayes_" + column] = naive_bayes_prediction_train
        combined_results_test["bayes_" + column] = naive_bayes_prediction_test

    ensemble = RandomForestClassifier()
    combined_results_test, combined_results_train = scale_features(combined_results_test, combined_results_train)
    ensemble.fit(combined_results_train, y_train)
    ensemble_predict_test = ensemble.predict(combined_results_test)
    ensemble_predict_train = ensemble.predict(combined_results_train)
    print_results_train_test("Ensemble RandomForestClassifier", y_train, y_test, ensemble_predict_train,
                             ensemble_predict_test)

    ensemble = GradientBoostingClassifier()
    ensemble.fit(combined_results_train, y_train)
    ensemble_predict_test = ensemble.predict(combined_results_test)
    ensemble_predict_train = ensemble.predict(combined_results_train)
    print_results_train_test("Ensemble GradientBoostingClassifier", y_train, y_test, ensemble_predict_train,
                             ensemble_predict_test)


if __name__ == "__main__":
    main()
