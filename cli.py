# Authors:
# Baruch Rutman
# Dor Rozenhak
# Roi Amzallag

import sys
import os
from os import path
from Models.ModelBuilder import ModelBuilder
from datetime import datetime
import joblib as jb

# https://stackoverflow.com/a/287944/2457002
class bcolors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m' # Green
    WARNING = '\033[93m' # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

menu = f'''{bcolors.HEADER}CLI Frontend (cli.py){bcolors.ENDC}
USAGE: python cli.py -train <training file path> [Optional arguments]
NOTE: Paths with spaces are must be quoted by double quotes e.g. "Path with spaces\\test.csv"
Required:
\t-train  <path to file> \t Path to training file (*.csv)
Optional without specific order:
\t-struct <path to file> \t Path to structure file (*.txt)
\t-test   <path to file> \t Path to test file (*.csv)
\t-out    <path to folder> Folder to place output files to (will default where training file is located)
\t-joblib [enable, purge]  Enable joblib support (and purge exisiting cache) (default is "disable")
\t-bins   [1..] \t\t The number of bins to discretized with (default is 5)
\t-gain   [0.0..] \t Minimum amount of gain for splitting (Decision Trees) (default is 0.3)
\t-leafs  [1..] \t\t Minimum amount of data samples for splitting (Decision Trees) (default is 0)
\t-neighbors [1...]\t Number of nearest neighbors (K-nearest neigbors) (Odd number, default is 5)
\t-clusters [1...]\t Number of clusters (Cluster algorithms) (default is 8)
Miscellaneous:
\t-help \t\t\t Shows this menu'''
err_help = "Type \"python cli.py -help\" to see help information"

# CLI FrontEnd
class ConsoleOutput:
    def error(self, msg):
        print(f"{bcolors.FAIL}ERROR: {msg} {bcolors.ENDC}")
        print(err_help)
        sys.exit(0) # exit with 1 annoys the IDE

    def __init__(self, argv):
        if len(argv) == 1 or "-help" in sys.argv:
            print(menu)
            sys.exit(0)

        self.argDict = {
            "-train": None,
            "-struct": None,
            "-test": None,
            "-out": None,
            "-bins": 5,
            "-gain": 0.3,
            "-leafs": 0,
            "-joblib": "disable",
            "-neighbors": 5,
            "-clusters": 8,
        }

        self.args, processed = argv[1:], 0
        while processed < len(self.args):
            if self.args[processed] not in self.argDict:
                self.error(f"Invalid command line argument: {self.args[processed]}")

            # Pull 2 at a time to process
            arg, val = self.args[processed], self.args[processed + 1]
            try:
                if arg in ["-bins", "-leafs", "-neighbors", "-clusters"]: # Integer values
                    try:
                        val = int(val)
                    except:  # Replace python-speak with custom message
                        raise ValueError(f"Invalid {arg} value")
                    if val <= 0:
                        raise ValueError(f"{arg} value must be 1 or higher")
                    if arg == "-neighbors" and val % 2 == 0:
                        raise ValueError(f"{arg} value must be odd")
                elif arg in ["-gain"]: # Floating point values
                    try:
                        val = float(val)
                    except: # Replace python-speak with custom message
                        raise ValueError(f"Invalid {arg} value")
                    if val < 0.0 or val == float("inf"):
                        raise ValueError(f"{arg} value must be 0.0 or higher")
                elif arg == "-joblib":
                    if val not in ["enable", "purge", "disable"]:
                        raise ValueError(f"{val} is an invalid {arg} value")
                elif arg in ["-out"]: # Folder path values
                    if path.isfile(val):
                        raise ValueError(f"{val} is not a valid folder path")
                else:
                    if not path.isfile(val): # File path values
                        raise ValueError(f"{val} is not a valid file path")
            except ValueError as ve:
                self.error(str(ve))

            self.argDict[arg] = val
            processed += 2

        if self.argDict["-train"] is None:
            self.error("No training file was specifed.")

        if self.argDict["-out"]:
            self.out_path = self.argDict["-out"]
            if not os.path.isdir(self.out_path):
                os.mkdir(self.out_path)
        else:
            self.out_path = path.split(path.abspath(self.argDict["-train"]))[0]

        self.log, self.overviewTable = [], None
        print(f"{bcolors.OKGREEN}Command line arguments successfully parsed{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Initializing primary model builder{bcolors.ENDC}")
        # Create the model builder object (which will do the other things automatically)
        self.modelBuilder = ModelBuilder(self.argDict["-train"],
                                         self.argDict["-struct"],
                                         self.argDict["-test"],
                                         self.argDict["-out"],
                                         self.argDict["-bins"],
                                         self.argDict["-gain"],
                                         self.argDict["-leafs"],
                                         self.argDict["-joblib"],
                                         self.argDict["-neighbors"],
                                         self.argDict["-clusters"],
                                         self)
    # Interface methods for ModelBuilder
    def logLine(self, line):
        line = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " " + line
        self.log.append(line)
        print(line)

    def addRow(self, row):
        # To be sorted out
        if self.overviewTable is None:
            self.overviewTable = row.to_frame().T
        else:
            self.overviewTable = self.overviewTable.append(row)

    def done(self):
        with open(path.join(self.out_path, "cli.log"), "w") as logfile:
            logfile.write("\n".join(self.log))
            print(f"{bcolors.OKGREEN}Log file written to {path.join(self.out_path, 'cli.log')}{bcolors.ENDC}")

        if self.overviewTable is not None:
            self.overviewTable.to_csv(path.join(self.out_path, "overview.csv"))
            print(f"{bcolors.OKGREEN}Overview table written to {path.join(self.out_path, 'overview.csv')}{bcolors.ENDC}")
            self.overviewTable.to_html(path.join(self.out_path, "overview.html"))
            print(f"{bcolors.OKGREEN}Overview table html version written to {path.join(self.out_path, 'overview.html')}{bcolors.ENDC}")

        with open(path.join(self.out_path, "execute.bat"), "w") as bat_file: # Unix support?
            bat_file.write(" ".join(["python", path.realpath(__file__)] + self.args))
            print(f"{bcolors.OKGREEN}Commandline copy saved to {path.join(self.out_path, 'execute.bat')}{bcolors.ENDC}")
        
if __name__ == "__main__":
    ConsoleOutput(sys.argv)
