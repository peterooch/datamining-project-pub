# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

import math
import pandas as pd

# Returns the amount of bits required to encode the given data set most effciently
# Lab 2
def Entropy(data):
    if type(data) != list:
        data = list(data)
    data_count = len(data)
    # Initialize count dictionary
    counts = {key: data.count(key) for key in set(data)}
    return -(sum([((counts[key]/data_count) * math.log2(counts[key]/data_count)) for key in counts]))

# E(A, T; S) -> CE(S, A, T)
def CutpointEntropy(df, attr, cutpoint):
    # Make this more "math" friendly
    S = df
    S1, S2 = S[S[attr] <= cutpoint], S[S[attr] > cutpoint]
    len_s, len_s1, len_s2 = len(S), len(S1), len(S2)
    return (len_s1 / len_s) * Entropy(S1["class"]) + (len_s2 / len_s) * Entropy(S2["class"])

# This function will always return an iterable, a list object for recursive calls, a pd.Series for main call
def EntropyDisc(df, attr, levels, init=True):
    if len(df) <= 1 or levels == 0:
        return [] if init is False else df[attr]

    # Make a sorted copy to work with but only really sort if needed
    df_orig, df = df, df.sort_values(by=[attr]) if init is True else df
    # Make sure we will not have redundant calculations
    unique_values = df[attr].unique()
    if len(unique_values) <= 1: # HACK Dunno if this works
        return [] if init is False else df[attr]
    # This is the greedy part of the algorithm FIXME Try to reduce amount of calculations
    entropies = pd.Series({value: CutpointEntropy(df, attr, value) for value in unique_values})
    # https://www.saedsayad.com/supervised_binning.htm
    cutpoint = entropies.idxmin() # Academic Paper said cut-point should be the lowest cutpoint entropy

    S1, S2 = df[df[attr] <= cutpoint], df[df[attr] > cutpoint]
    cutpoints = EntropyDisc(S1, attr, levels - 1, False) + [cutpoint] + EntropyDisc(S2, attr, levels - 1, False)

    if init is not True:
        return cutpoints
    # Add maximum and minimum values so not to miss any values
    cutpoints += [unique_values.min(), unique_values.max()]
    # Have pandas.cut to actually do the binning for us
    return pd.cut(df_orig[attr], sorted(cutpoints), duplicates="drop", include_lowest=True)
