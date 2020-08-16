# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from Utilities.PDUtils import SplitXY, IsInterval, FindBestInterval

class Type2Tree:
    def __init__(self, df, **kwargs):
        x, y = SplitXY(df)
        self.common_class = y.value_counts().idxmax()
        if "encoder" not in kwargs or kwargs["encoder"] is None:
            self.encoder = OrdinalEncoder().fit(x)
        else:
            self.encoder = kwargs["encoder"]
        leaf_limit = 2 if "leaf_limit" not in kwargs or kwargs["leaf_limit"] < 2 else kwargs["leaf_limit"]
        self.model = DecisionTreeClassifier(criterion="entropy", min_samples_split=leaf_limit)
        self.model.fit(self.encoder.transform(x), y)

    def evaluate(self, row):
        row = row.drop(labels=["class"])
        for label, intervals in zip(row.index, self.encoder.categories_):
            if IsInterval(intervals[0]):
                row[label] = FindBestInterval(row[label], intervals)
        try:
            return self.model.predict(self.encoder.transform([row]))[0]
        except Exception:
            return self.common_class
