import warnings;

warnings.filterwarnings("ignore")
import pandas as pd


def print_statistics(frame):
    print("Columns:",frame.columns)
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
    frame = frame.astype({'class': int})
    frame.rename(index=str, columns={'Unnamed: 0': 'number', 'class': 'result'},inplace=True)
    print("Columns:",frame.columns)
    frame.set_index('number',inplace=True)
    return frame



# General initialization
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
csv_filename = "gold_data.csv"

df = read_data(csv_filename)
print_statistics(df)
