import pandas as pd
import Discretizators as disc
ENABLE_TEST1 = False
ENABLE_TEST2 = True

# Test to make sure that some functions work

if ENABLE_TEST1:
    df = pd.read_csv("train.csv")

    df.sort_values(by=["age"], inplace=True)
    diff = df["age"].max() - df["age"].min()
    n = 6
    sample_vals = [df["age"].min() + i * (diff / n) for i in range(1, n)]

    for val in sample_vals:
        print("{0:.2f} {1}".format(val, disc.CutpointEntropy(df, "age", val)))

if ENABLE_TEST2:
    df1 = pd.DataFrame({"num":[4,5,8,12,15], "class": ['N','Y','N','Y','Y']})
    df2 = pd.DataFrame({"num":[8,15,12,4,5], "class": ['N','Y','Y','N','Y']})
    print("df1\n", disc.EntropyDisc(df1, "num", 5))
    print("df2\n", disc.EntropyDisc(df2, "num", 5))
