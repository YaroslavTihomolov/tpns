import numpy as np
from pandas import DataFrame


class GainRatio:
    def __init__(self, df: DataFrame, target: str):
        self.df = df
        self.target = target

    def entropy(self, df_part, attribute):
        entropy_value = 0
        summary_count = df_part[attribute].count()
        for i in df_part[self.target].value_counts():
            peace = i / summary_count
            entropy_value -= peace * np.log2(peace)
        return entropy_value

    def information_gain(self, column):
        ig = self.entropy(self.df, self.target)
        summary_count = self.df[self.target].count()
        for i in self.df[column].unique():
            value_df = self.df[self.df[column] == i]
            value_count = value_df[column].count()
            ig -= (value_count / summary_count) * self.entropy(value_df, column)
        return ig

    def intrinsic_information(self, column):
        ii = 0
        summary_count = self.df[self.target].count()
        for i in self.df[column].value_counts():
            peace = i / summary_count
            ii -= peace * np.log2(peace)
        return ii

    def gain_ratio(self):
        results = []
        for i in self.df.columns:
            if i == self.target:
                continue
            ig = self.information_gain(i)
            ii = self.intrinsic_information(i)
            gain_ratio_val = ig / ii
            results.append((i, gain_ratio_val))
        return results

