# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

import pandas as pd
from pandas.api.types import is_numeric_dtype
from functools import reduce
import Utilities.PDUtils as utils

class Type1Bayes:
    "Self implemented Naive Bayes Classifier Model"
    def __init__(self, df, **kwargs):
        # Dictionaries to store models and memoized calculations
        self.probTables, self.len_cache, self.prob_cache = dict(), dict(), dict()
        self.classes = df["class"].unique()

        if "encoder" not in kwargs or kwargs["encoder"] is None:
            for col in df:
                if col == "class":
                    continue
                self.probTables[col] = self.parseColumn(df, col)
        else:
            for col, categories in zip(df, kwargs["encoder"].categories_):
                if col == "class":
                    continue
                self.probTables[col] = self.parseColumn(df, col, categories)

        self.probTables["class"] = pd.DataFrame(1, columns=["value"], index=self.classes)
        for class_value in df["class"]:
            self.probTables["class"].at[class_value, "value"] += 1

    # df should have been processed by CleanDataFrame
    def parseColumn(self, df, col_name, categories = None):
        # put 1 as a default value as laplacian correction
        if categories is not None:
            result = pd.DataFrame(1, columns=categories, index=self.classes)
        else:
            result = pd.DataFrame(1, columns=df[col_name].unique(), index=self.classes)
        for class_value, col_value in zip(df["class"], df[col_name]):
            result.at[class_value, col_value] += 1
        return result

    def probabilityOf(self, table, a, b=None):
        # Use the caches to avoid wasting time on recalculating already calculated values
        if (table, a, b) not in self.prob_cache:
            if table not in self.len_cache:
                self.len_cache[table] = sum([sum(self.probTables[table][col]) for col in self.probTables[table]])
            try:
                if b is None:
                    self.prob_cache[(table, a, b)] = sum(self.probTables[table].loc[a]) / self.len_cache[table]
                else:
                    self.prob_cache[(table, a, b)] = (self.probTables[table].loc[b, a] / self.len_cache[table]) / self.probabilityOf(table, b)
            except Exception:
                self.prob_cache[(table, a, b)] = 1
        return self.prob_cache[(table, a, b)]

    # Row eval function
    def evaluate(self, row):
        probabilities = {c: self.probabilityOf("class", c) for c in self.classes}
        for key in row.index:
            if key == "class":
                continue
            if row[key] not in self.probTables[key] and is_numeric_dtype(row[key]):
                # Find the column that fits the row value
                col = utils.FindBestInterval(row[key], self.probTables[key])
            else:
                col = row[key]
            for c in self.classes:
                probabilities[c] *= self.probabilityOf(key, col, c)
        # find class with highest probability and return it
        return reduce(lambda a, b: a if probabilities[a] > probabilities[b] else b, probabilities)
