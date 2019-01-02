import math
import re
import string
import pandas as pd


class BayesDetector(object):
    """Implementation of Naive Bayes for binary classification"""
    def __init__(self,column):
        self.column = column

    def positive_to_boolean(self, result: list):
        res = pd.Series(result)
        res = res.where(res < 0, 1)
        res = res.where(res >= 0, 0)
        return res

    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        remove_digits = str.maketrans('', '', string.digits)
        return s.translate(remove_digits).translate(translator)

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def get_n_longest_values(self, dictionary, n):
        longest_entries = sorted(
            dictionary.items(), key=lambda t: t[1], reverse=True)[:n]
        dict_return_value = [(key, value) for key, value in longest_entries]
        return dict(dict_return_value)

    def fit(self, X_df, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        X = X_df[self.column]
        n = len(X)
        self.num_messages['Acquisition'] = sum(1 for label in Y if label == 1)
        self.num_messages['NoAcquisition'] = sum(1 for label in Y if label == 0)
        self.log_class_priors['Acquisition'] = math.log(self.num_messages['Acquisition'] / n)
        self.log_class_priors['NoAcquisition'] = math.log(self.num_messages['NoAcquisition'] / n)
        self.word_counts['Acquisition'] = {}
        self.word_counts['NoAcquisition'] = {}

        for x, y in zip(X, Y):
            c = 'Acquisition' if y == 1 else 'NoAcquisition'
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count

        self.word_counts["Acquisition"] = self.get_n_longest_values(self.word_counts["Acquisition"], 100)
        self.word_counts["NoAcquisition"] = self.get_n_longest_values(self.word_counts["NoAcquisition"], 100)
        self.vocab = set(self.word_counts["Acquisition"].keys()).union(self.word_counts["NoAcquisition"].keys())
        return

    def prediction_and_confidence(self, X_df):
        X = X_df[self.column]
        result = []
        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            acquisition_score = 0
            no_acquisition_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue

                # add Laplace smoothing
                log_w_given_acquisition = math.log(
                    (self.word_counts['Acquisition'].get(word, 0.0) + 1) / (
                            self.num_messages['Acquisition'] + len(self.vocab)))
                log_w_given_no_acquisition = math.log(
                    (self.word_counts['NoAcquisition'].get(word, 0.0) + 1) / (
                            self.num_messages['NoAcquisition'] + len(self.vocab)))

                acquisition_score += log_w_given_acquisition
                no_acquisition_score += log_w_given_no_acquisition

            acquisition_score += self.log_class_priors['Acquisition']
            no_acquisition_score += self.log_class_priors['NoAcquisition']
            result.append(acquisition_score - no_acquisition_score)

        result = pd.Series(result)
        result_int = self.positive_to_boolean(result)
        return result_int, result
