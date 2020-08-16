Authors:
    - Baruch Rutman
    - Dor Rozenhak
    - Roi Amzallag

Required libraries:
    - pandas
      used for csv file parsing, data handling, data manipulation and data storage
    - sklearn
      used in KMEANS, KNN, Type2Bayes and Type2Tree models
      used for calculation and generation of Confusion Matrix PDFs
    - joblib
      used for effcient on-disk storage of objects, to reduce compute time and reuse of models after generation
    - matplotlib
      used for generation of Confusion Matrix PDFs

The program is CLI Frontend to the ModelBuilder object that does all the model generation and evaluation.

How to use (copied from cli.py):

How to see this text via the program: >>> python cli.py -help

USAGE: python cli.py -train <training file path> [Optional arguments]
NOTE: Paths with spaces are must be quoted by double quotes e.g. "Path with spaces\test.csv"
Required:
    -train  <path to file>      Path to training file (*.csv)
Optional without specific order:
    -struct <path to file>      Path to structure file (*.txt)
    -test   <path to file>      Path to test file (*.csv)
    -out    <path to folder>    Folder to place output files to (will default where training file is located)
    -joblib [enable, purge]     Enable joblib support (and purge exisiting cache) (default is "disable")
    -bins   [1..]               The number of bins to discretized with (default is 5)
    -gain   [0.0..]             Minimum amount of gain for splitting (Decision Trees) (default is 0.3)
    -leafs  [1..]               Minimum amount of data samples for splitting (Decision Trees) (default is 0)
    -neighbors [1...]           Number of nearest neighbors (K-nearest neigbors) (Odd number, default is 5)
    -clusters [1...]            Number of clusters (Cluster algorithms) (default is 8)
Miscellaneous:
    -help                       Shows this menu

So if you would want to use the program with the following parameters:
  Training file - train.csv
  Test file - test.csv
  Structure file - Struct.txt
  Joblib - Enabled
  Folder location for output files - C:/output
  Bin count for continuos features - 5
  Minimum gain for tree algorithms - 0.2
  Leaf limit for branching for tree algorithms - 25
  Neighbor count for K-nearest neighbors algorithms - 5
  Cluster count for cluster algorithms - 10

The relevant commandline to achieve all of the above would be
python cli.py -train train.csv -test test.csv -struct Struct.txt -joblib enable -out C:/output -bins 5 -gain 0.2 -leafs 25 -neigbors 5 -clusters 10

The program will automatically will generate the models, will test them against training data before discretization,
training data used for building the model (after disc), and the test file if specified.

It will record the results and save the results as files in the output folder, with the models stored at the "models" subfolder (if joblib is enabled), 
per-line results stored in csv files and Confusion Matrix pdf based on those per-line samples.
In addition a cleaned copy of the training file will be saved and a processed version of the training data per discretization method.
Also a general summary data will be save both in csv and html versions in addition batch file with copy of the commandline so the results can be replicated at will.
And if joblib is enabled then the ModelBuilder object itself will be stored and can be iterated if a new test set needs to be tested on the models. 

for model in modelBuilder:
    <do stuff with model>

If a model needed to be used after generation, each model has an inteface method that works like this:
ModelObject.evaluate(<data row without classification>) => result: a classification determined upon how the model is built
