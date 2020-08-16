# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

# dor responcebilty
import math

from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from Utilities.PDUtils import IsInterval, FindBestInterval, SplitXY

class KNN:
    "KNN Model implemented with SKLearn classes"
    def __init__(self , df , **kwargs):
        #split data
        x, y = SplitXY(df)
        self.common_class = y.value_counts().idxmax()
        if "encoder" not in kwargs or kwargs["encoder"] is None:
            self.encoder = OrdinalEncoder().fit(x)
        else:
            self.encoder = kwargs["encoder"]

        Nneighbors = 5 if "neighbors" not in kwargs else kwargs["neighbors"]
        self.classifier = KNeighborsClassifier(n_neighbors=Nneighbors, metric = 'euclidean')
        self.classifier.fit(self.encoder.transform(x),y)

    def evaluate(self, row): # "yes" | "no" | None
        row = row.drop(labels=["class"])
        for label, intervals in zip(row.index, self.encoder.categories_):
            if IsInterval(intervals[0]):
                row[label] = FindBestInterval(row[label], intervals)
        try:
            return self.classifier.predict(self.encoder.transform([row]))[0]
        except Exception:
            return self.common_class
