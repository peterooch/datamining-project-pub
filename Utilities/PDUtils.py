# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder
from Utilities.Discretizators import EntropyDisc

__doc__ = '''Utility methods for dealing with Pandas ordered data (DataFrame, Series, Interval)'''

# Imported from lab2 work with minor adaption for usage with pandas series
def Entropy(data):
    '''
    Returns the amount of bits required to encode the given data set most effciently\n
    data - assumed to be pandas.Series object
    '''
    # Initialize count dictionary
    counts = {key: data.value_counts()[key] for key in data.unique()}
    return -(sum([((counts[key]/len(data)) * math.log2(counts[key]/len(data))) for key in counts]))

# https://stackoverflow.com/questions/33468976/pandas-conditional-probability-of-a-given-specific-b
def Gain(data, attribute):
    '''
    Returns the information gain for the given attribute based upon the given data\n
    data - assumed to be pandas.DataFrame object\n
    attribute - assumed to be a column header in data, scalar value
    '''
    result = Entropy(data["class"])
    combinations = data.groupby(attribute)["class"].value_counts()
    attr_vals = data.groupby(attribute)["class"].count()
    probs = combinations / attr_vals
    for entry in attr_vals.index:
        entropy = 0
        for combination in [cb for cb in combinations.index if entry in cb]:
            entropy += (probs[combination]) * math.log2(probs[combination])
        result += (attr_vals[entry] / len(data)) * entropy
    return result

# Heuristically determine which column is numeric or discrete
# Less heuristic if a structure file is supplied
def CleanDataFrame(df, bin_count, disc_type = "equal depth", struct_file = None):
    # Make everything lower case
    df = df.applymap(lambda x : x.lower() if type(x) == str else x)

    if struct_file is not None:
        value_matrix, col_list = StructureFileParser(struct_file)
        #Remove any rows that dont have any classification or bogus ones
        classes = value_matrix["class"]
        del value_matrix["class"]
        df = df[[item in classes for item in df["class"]]]
        for col in df:
            if col not in col_list:
                df = df.drop(labels=col, axis=1)

    for col in df:
        if is_numeric_dtype(df[col]):
            # Replace NaNs with mean value
            df[col].fillna(df[col].mean(), inplace = True)
            if bin_count < 1:
                continue
            # Bin the values
            if disc_type == "equal depth":
                bins = pd.qcut(df[col], bin_count, duplicates="drop")
            elif disc_type == "equal width":
                bins = pd.cut(df[col], bin_count, duplicates="drop")
            elif disc_type == "entropy":
                bins = EntropyDisc(df, col, max(int(math.log2(bin_count)),2))
            else:
                if struct_file is not None and value_matrix[col] is None:
                    value_matrix[col] = sorted(df[col].unique())
                continue # Dont do discretization on this
            # Replace each cell with its bin object
            for i in range(len(df[col])):
                df.at[i, col] = bins[i]
            if struct_file is not None and value_matrix[col] is None:
                value_matrix[col] = sorted(set(bins))
        else:
            # Fill empty cells with the most common value
            df[col].fillna(df[col].value_counts().idxmax(), inplace = True)
    if struct_file is None:
        return df, None
    else: # Create a an ordinal encoder object for use with SKLearn classes
        return df, OrdinalEncoder(categories=list(value_matrix.values())).fit(SplitXY(df)[0])

def StructureFileParser(struct_file):
    mtx = dict()
    cols = []
    with open(struct_file) as file:
        struct_lines = file.read().split("\n")
    
    for line in struct_lines:
        segments = line.split()
        cols.append(segments[1])
        discrete_vals = segments[2].strip("{ }")
        if discrete_vals == "NUMERIC":
            mtx[segments[1]] = None
        else:
           mtx[segments[1]] = discrete_vals.split(",")
    return mtx, cols

def FindBestInterval(value, intervals):
    '''Find the most fitting interval for a value, assumes no gaps between the intervals and no overlaps'''
    min_interval = max_interval = intervals[0]

    if IsInterval(value):
        return value

    for interval in intervals:
        if value in interval:
            return interval
        if interval.right < min_interval.right:
            min_interval = interval
        if interval.left > max_interval.left:
            max_interval = interval
    
    return min_interval if min_interval.right >= value else max_interval

def IsInterval(value):
    '''Returns True if value is an interval, False otherwise'''
    return isinstance(value, (pd.Interval, pd.IntervalIndex))

def SplitXY(df):
    '''Split the df into 2 parts, useful for sklearn classes,\n
       X contains everything but the "class" column\n
       and Y contains the "class" column only.\n
       returns (X,Y)'''
    x = df.drop(labels=["class"], axis=1)
    y = df["class"]
    return x, y
