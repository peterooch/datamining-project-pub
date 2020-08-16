# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

from functools import reduce
import Utilities.PDUtils as utils

# Authors: Baruch Rutman and Roi Amzallag
# Imported lab5 ID3 tree builder

class Type1ID3Tree:
    '''
    Self implemented ID3 Tree Model\n
    ID3 Decision tree class\n
    Built as a recursive tree with the leaves being one of the possible class results
    '''
    # Minimum info gain value, to not over fit the model, usually 0.3
    MIN_GAIN = 0.3
    def __init__(self, df, **kwargs):
        '''
        DecisionTree Construtor\n
        df - df is assumed to be a pandas.DataFrame object, might have bugs if not\n
        leaf_limit - set so to not have more branching if len(df) <= leaf_limit\n
        skip_attrs - list of attributes that are not be evaluated for InfoGain\n
        min_gain - minimum amount of InfoGain for branching, default is 0.3  
        '''
        leaf_limit = 0 if "leaf_limit" not in kwargs else kwargs["leaf_limit"]
        skip_attrs = ["class"] if "skip_attrs" not in kwargs else kwargs["skip_attrs"]
        min_gain = MIN_GAIN if "min_gain" not in kwargs else kwargs["min_gain"]

        # Save the structure for use later in evaluation
        if "attr_dict" not in kwargs:
            self.attr_dict = dict()
            temp_df = df.drop(labels="class", axis=1)
            if "encoder" in kwargs and kwargs["encoder"] is not None:
                for attr, categories in zip(temp_df, kwargs["encoder"].categories_):
                    self.attr_dict[attr] = categories
            else:
                for attr in temp_df:
                    self.attr_dict[attr] = sorted(temp_df[attr].unique())
        else:
            self.attr_dict = kwargs["attr_dict"]

        self.subTrees, self.rootAttr = None, None
        # Set most common class value for this sub tree
        val_counts = df["class"].value_counts()
        self.classResult = val_counts.idxmax()
        # Filter out attributes that are to be skipped
        attrs = [attr for attr in df if attr not in skip_attrs]
        # Check if there attrs to work with or check if data len is below the set threshhold
        if len(attrs) == 0 or (leaf_limit > 0 and len(df) < leaf_limit) or len(val_counts) <= 1:
            return
        # Calculate the gain for each attribute
        #gains = {attr: utils.Gain(df, attr) for attr in attrs}
        # FIXME Maybe this instead?
        gains = dict()
        for attr in attrs:
            gains[attr] = 0
            entropy = utils.Entropy(df[attr])
            if entropy != 0:
                gains[attr] = utils.Gain(df,attr) / entropy
        # Pick the attribute to branch on
        self.rootAttr = reduce(lambda a, b: a if gains[a] > gains[b] else b, gains)
        # Check if the highest is below the set threshhold
        if gains[self.rootAttr] <= min_gain:
            return
        self.subTrees, unique_values = dict(), df[self.rootAttr].unique()
        for entry in unique_values:
            # Branch and take only the relevant rows (divide and conquer)
            self.subTrees[entry] = Type1ID3Tree(df[df[self.rootAttr] == entry],
                                                leaf_limit = leaf_limit,
                                                skip_attrs = skip_attrs + [self.rootAttr],
                                                min_gain = min_gain,
                                                attr_dict = self.attr_dict)
        if len(self.subTrees) == 1:
            # Pruning
            subTree = self.subTrees[list(self.subTrees.keys())[0]]
            self.classResult, self.rootAttr, self.subTrees = subTree.classResult, subTree.rootAttr, subTree.subTrees
    # Row eval function
    def evaluate(self, row, first_call = True):
        if first_call: # Reform row according to training set structure
            row = row.drop(labels=["class"])
            for label in row.index:
                if utils.IsInterval(self.attr_dict[label][0]):
                    row[label] = utils.FindBestInterval(row[label], self.attr_dict[label])
        if self.subTrees is not None and row[self.rootAttr] in self.subTrees:
            return self.subTrees[row[self.rootAttr]].evaluate(row, first_call = False)
        # return most common class value
        return self.classResult
