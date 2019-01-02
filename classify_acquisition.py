import itertools
import pickle
import os.path

import pandas as pd
from answer_classifier import answer_classifier


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

    print("\r\n\r\n*** Test method classify() on some positive examples ***")
    df_filtered = df[df['result'] == 1]
    num_rows = 4
    for index, row in itertools.islice(df_filtered.iterrows(), num_rows):
        print("row: ", index)
        url = row["url"]
        title = row["title"]
        description = row["description"]
        content = row["content"]
        prediction = answer.classify(url, title, description, content)
        print("{0}:\r\n{1}\r\n{2}:\r\n{3}\r\n{4}:\r\n{5}\r\n{6}:\r\n{7}\r\n{8}{9}\r\n{10}{11}\r\n----------------".format(
            "url",url,"title",title,"description",description,"content",content,"result:",row["result"],"predicted:",prediction
        ))

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
