# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

from functools import reduce
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from Utilities.PDUtils import FindBestInterval, IsInterval, SplitXY

# Roi's responsibility
class Type2Bayes:
    "Naive Bayes Classifier Model implemented with SKLearn classes"
    def __init__(self, df, **kwargs):
        x, y = SplitXY(df)
        if "encoder" not in kwargs or kwargs["encoder"] is None:
            self.encoder = OrdinalEncoder().fit(x)
        else:
            self.encoder = kwargs["encoder"]
        self.model = CategoricalNB()
        self.model.fit(self.encoder.transform(x), y)
    def evaluate(self, row):
        row = row.drop(labels=["class"])
        for label, intervals in zip(row.index, self.encoder.categories_):
            if IsInterval(intervals[0]): 
                row[label] = FindBestInterval(row[label], intervals)
        try:
            return self.model.predict(self.encoder.transform([row]))[0]
        except Exception:
            # return most common class
            return reduce(lambda x, y: x if x[1] > y[1] else y, zip(self.model.classes_, self.model.class_count_))[0]
