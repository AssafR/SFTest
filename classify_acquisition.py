import warnings;
import pickle
import os.path

warnings.filterwarnings("ignore")
import pandas as pd
import pre_processor


def print_statistics(frame):
    print("Columns:", frame.columns)
    print("Head:")
    print(frame.head())
    print("Description:")
    print(frame.describe())
    counts = pd.Series.value_counts(frame['result'])
    print("Result frequency:")
    print(counts)


def read_data(filename):
    frame = pd.read_csv(filename)
    frame = frame.dropna(axis=0, subset=['class'])
    frame = frame.astype({'class': int, 'title':str})
    frame.rename(index=str, columns={'Unnamed: 0': 'number', 'class': 'result'}, inplace=True)
    print("Columns:", frame.columns)
    frame.set_index('number', inplace=True)
    return frame


def contains(substring: str, string: str):
    return int(substring in string)


# General initialization
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
csv_filename = "gold_data.csv"
pickle_filename = "data.pkl"

if not os.path.isfile(pickle_filename):
    df = read_data(csv_filename)
    pickle.dump(df, open(pickle_filename, "wb"))
else:
    df = pickle.load(open(pickle_filename,"rb"))

df['title_contains_acquisition'] = df["title"].apply(lambda x: contains("acquisition", x))
print_statistics(df)
