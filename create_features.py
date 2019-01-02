import pandas as pd


def contains(substring: str, string: str):
    return int(substring in string)


def count_occurences(substring: str, string: str):
    return string.count(substring)


def create_new_count_feature(df: pd.DataFrame, column: str, substring: str, suffix=None):
    suffix = suffix if suffix else substring
    new_column_name = column + "_contains_" + suffix
    df[new_column_name] = df[column].apply(lambda x: count_occurences(substring, x))


def create_new_count_features(df: pd.DataFrame, columns, dct: dict):
    for column in columns:
        for substring, suffix in dct.items():
            create_new_count_feature(df, column, substring, suffix)


def create_new_features(df: pd.DataFrame):
    old_columns = set(df.columns)
    # url, title, description, content
    # Hand-crafted features
    create_new_count_features(df, ("title", "description", "content"),
                              {"acquisition": None, "acquires": None, "acquired": None})
    # Features: Url site, URL

    new_columns = list(set(df.columns) - old_columns)
    result = pd.DataFrame(df[new_columns])
    df.drop(columns=new_columns, inplace=True)
    return result
