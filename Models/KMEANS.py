# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from Utilities.PDUtils import IsInterval, FindBestInterval, SplitXY

class KMEANS:
    "KMeans Model implemented with SKLearn classes"
    def __init__(self, df, **kwargs):
        x, y = SplitXY(df)
        clus_count = 8 if "cluster" not in kwargs else kwargs["cluster"]
        if "encoder" not in kwargs or kwargs["encoder"] is None:
            self.encoder = OrdinalEncoder().fit(x)
        else:
            self.encoder = kwargs["encoder"]

        self.common_class = y.value_counts().idxmax()
        self.model = KMeans(n_clusters = clus_count)
        self.model.fit(self.encoder.transform(x))
        # The clusterID -> class label as requested in the assignment document
        # Prepare placeholder df to store temporary counts
        cls_count_df = pd.DataFrame(0, index=y.unique(), columns=pd.RangeIndex(clus_count))
        # Iterate on the labels in the model and the dataframe
        for clus_idx, row_class in zip(self.model.labels_, y):
            cls_count_df.at[row_class, clus_idx] += 1
        # Pick class value with highest count
        self.cluster_classes = [cls_count_df[col].idxmax() for col in cls_count_df]
    def evaluate(self, row):
        row = row.drop(labels=["class"])
        for label, intervals in zip(row.index, self.encoder.categories_):
            if IsInterval(intervals[0]):
                row[label] = FindBestInterval(row[label], intervals)
        try:
            clus_idx = self.model.predict(self.encoder.transform([row]))[0]
            return self.cluster_classes[clus_idx]
        except Exception:
            # Return most common class
            return self.common_class
