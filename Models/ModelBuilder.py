# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

# Library imports
from itertools import product
import os
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Internal function and model imports
from Utilities.PDUtils import CleanDataFrame
from Models.Type2Tree import Type2Tree
from Models.Type1ID3Tree import Type1ID3Tree
from Models.Type1Bayes import Type1Bayes
from Models.Type2Bayes import Type2Bayes
from Models.KNN import KNN
from Models.KMEANS import KMEANS

# Backend model generation and testing,
# can be used as stand alone class or with the CLI (cli.py) or GUI
#
# output_gui interface methods:
# output_gui.logLine(line: str) : status updates
# output_gui.addRow(row: pandas.Series) : add row to overview table
# row - [description, total, correct, errors, %]
# output_gui.done() : notify that all tasks are finished
class ModelBuilder:
    # Types should have a common builder definition and interface methods
    # So if we add a new model it would be a simple line addition to this dictionary and a import statement
    # Model(df, **kwargs) <- each constructor will take the relevant arguments from the kwargs
    # Model.evaluate(row) <- receives a row from a data set, including the class column
    #                        and returns a classification based upon the row data (excluding the class attribute)
    # Generated models will be made as cartesian product of DiscTypes and ModelTypes
    ModelTypes = {
        # Description: [Class]
        "Type1 NBC model, {0} disc": Type1Bayes, # Lab 4 Naive Bayes classifier
        "Type2 NBC model, {0} disc": Type2Bayes, 
        "Type1 Tree model, {0} disc": Type1ID3Tree, # Lab 5 ID3 classifier
        "Type2 Tree model, {0} disc": Type2Tree,
        "KNN model, {0} disc": KNN,
        "KMEANS model, {0} disc": KMEANS,
    }
    DiscTypes = ("equal width", "equal depth", "entropy")
    # Might more params in the future
    def __init__(self,
                 train_file,
                 struct_file = None,
                 test_file = None,
                 output_dir = None,
                 bin_count = 5,
                 min_gain = 0.3,
                 leaf_limit = 0,
                 jb_status = "disable",
                 neighbors = 5,
                 clusters = 8,
                 output_gui = None):
        # Set up general variables
        self.gui, self.jb_status, self.memory, self.models = output_gui, jb_status, None, []
        self.write_dir = output_dir if output_dir is not None else path.split(path.abspath(train_file))[0]
        self.gui_exists = output_gui is not None
        # Memoize CleanDataFrame
        self.CleanDataFrame = self.memoize(CleanDataFrame)

        self.dataframes, self.encoders, self.specifics = dict(), dict(), dict()
        # Initial cleanup
        self.log("Cleaning up training data")
        train_df, _ = self.CleanDataFrame(pd.read_csv(train_file), -1, "none")
        self.TagDataFrame(train_df, "train data without discretization")
        train_df.to_csv(self.GetPath("Training file cleaned.csv"))

        for disc in self.DiscTypes:
            self.log(f"Starting pre-processing with {disc} discretization")
            self.dataframes[disc], self.encoders[disc] = self.CleanDataFrame(train_df, bin_count, disc, struct_file)
            self.dataframes[disc].to_csv(self.GetPath(f"Training file processed using {disc} disc.csv"))
            self.TagDataFrame(self.dataframes[disc], f"train data with {disc} disc")
            self.log(f"Finished pre-processing with {disc} discretization")

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            self.TagDataFrame(test_df, "test data")

        self.log("Starting model runs...")
        for modelEntry, modelDisc in product(self.ModelTypes, self.DiscTypes):
            modelDesc = modelEntry.format(modelDisc)
            self.log(f"Building {modelDesc}")

            # Add the aditional params here if they are added to __init__
            modelObj = self.CreateModel(modelEntry,
                                        modelDisc,
                                        min_gain = min_gain,
                                        leaf_limit = leaf_limit,
                                        neighbors = neighbors,
                                        clusters = clusters,
                                        encoder = self.encoders[modelDisc])

            self.log(f"Finished building {modelDesc}")
            # Evaluate data
            self.evaluate_df(self.dataframes[modelDisc], modelObj, f"{modelDesc}, processed data")
            self.evaluate_df(train_df, modelObj, f"{modelDesc}, unprocessed data")
            if test_file is not None:
                self.evaluate_df(test_df, modelObj, f"{modelDesc}, test data")

            # Keep object
            self.models.append(modelObj)
        self.log("Generating confusion matrix pdfs")
        for df_ident in self.specifics:
            df = self.specifics[df_ident] # Simplify code a bit
            df.to_csv(self.GetPath(f"Per sample results - {df_ident}.csv"))
            y_true = df["class"]
            y_preds = [df[col] for col in df.drop(labels=["class"], axis=1)]
            classes = y_true.unique()
            # Inspired by https://stackoverflow.com/questions/59165149/plot-confusion-matrix-with-scikit-learn-without-a-classifier
            # And https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
            cols, rows = len(self.DiscTypes), int(np.ceil(len(y_preds) / len(self.DiscTypes)))
            fig, axes = plt.subplots(rows, cols, squeeze=False)
            axes_list = []
            for items in axes:
                axes_list += [*items]
            for y_pred, ax in zip(y_preds, axes_list):
                mtx = confusion_matrix(y_true, y_pred, labels=classes)
                ConfusionMatrixDisplay(mtx, display_labels=classes).plot(ax=ax, cmap=plt.cm.Blues)
                ax.set_title(y_pred.name)
                ax.set_ylabel("Expected class")
                ax.set_xlabel("Predicted class")
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            fig.set_size_inches(cols * 5.2, rows * 4.1)
            fig.savefig(self.GetPath(f"Confusion Matrix - {df_ident}.pdf"))
        self.log("Finished generation of confusion matrix pdfs")

        if self.gui_exists:
            # GUI/CLI should display/dump results into the requested medium
            self.gui.done()
        if self.jb_status in ["enable", "purge"]:
            # Save self as a bunched up collection of data and models
            # Avoid pesky issue with missing gui
            builder_path = self.GetPath("ModelBuilder.joblib")
            self.log(f"Saving ModelBuilder object to {builder_path}")
            self.gui, self.gui_exists = None, False
            jb.dump(self, builder_path)

    def CreateModel(self, modelEntry, modelDisc, **kwargs):
        modelDesc = modelEntry.format(modelDisc)
        if self.jb_status != "enable":
            return self.ModelTypes[modelEntry](self.dataframes[modelDisc], **kwargs)
        if not os.path.isdir(self.GetPath("models")):
            os.mkdir(self.GetPath("models"))
        file_path = self.GetPath(f"models/{modelDesc}.joblib")
        try:
            # Check if model already exists
            model = jb.load(file_path)
            self.log(f"Loaded model {modelDesc} from joblib file.")
        except Exception:
            model = self.ModelTypes[modelEntry](self.dataframes[modelDisc], **kwargs)
            jb.dump(model, file_path)
            self.log(f"Saved model {modelDesc} to joblib file.")
        model.description = modelDesc # Have this saved for __iter__
        return model

    def evaluate_df(self, df, modelObj, modelDesc):
        self.log(f"Starting evaluation of {modelDesc}")
        df_ident = df.ident
        df = df.applymap(lambda x: x.lower() if type(x) == str else x)
        info = {"Total Entries": len(df), "Correct": 0, "Errors": 0, "Error %": 0}
        
        specifics = pd.Series(index=range(len(df)), name=modelDesc, dtype=object)
        for idx, row in df.iterrows():
            result = specifics[idx] = modelObj.evaluate(row)
            if result is None or result != row["class"]:
                info["Errors"] += 1
            else:
                info["Correct"] += 1

        self.specifics[df_ident][modelDesc] = specifics
        info["Error %"] = format((info["Errors"] / len(df)) * 100, '.2f')
        self.log(", ".join([f"{key}: {info[key]}" for key in info]))
        self.log(f"Finished evaluation of {modelDesc}")
        # Send Result to relevant GUI
        if self.gui_exists:
            self.gui.addRow(pd.Series(info, name=modelDesc))

    def log(self, line):
        if self.gui_exists:
            self.gui.logLine(line)
    def __iter__(self):
        return iter(self.models)
    # Trade compute time for harddrive space
    def memoize(self, func):
        if self.jb_status not in ["enable", "purge"]:
            return func
        if self.memory is None:
            self.memory = jb.Memory(self.write_dir, verbose=0)
            if self.jb_status == "purge":
                self.memory.clear(warn=False)
                self.jb_status = "enable" # Prevent purge on every call
        return self.memory.cache(func)
    def GetPath(self, fname):
        return os.path.join(self.write_dir, fname)
    def TagDataFrame(self, df, ident):  
        df.ident = ident
        self.specifics[ident] = df["class"].to_frame()
