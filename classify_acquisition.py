import pickle
import os.path

import pandas as pd
import create_features
from answer_classifier import answer_classifier
from bayes_detector import BayesDetector


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


def main():
    df = read_data()
    answer = answer_classifier(df)
    answer.run_train_test(df)



if __name__ == "__main__":
    main()
