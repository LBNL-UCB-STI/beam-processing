import os

import pandas as pd


class InputDirectory:
    def __init__(self, path: str):
        self.directoryPath = path
        self.isLink = "://" in path

    def append(self, relativePath):
        if self.isLink:
            if type(relativePath) in [list, tuple]:
                return "/".join([self.directoryPath] + relativePath)
            else:
                return "/".join([self.directoryPath, relativePath])
        else:
            if type(relativePath) in [list, tuple]:
                return os.path.join(self.directoryPath, *relativePath)
            elif type(relativePath) is str:
                return os.path.join(self.directoryPath, relativePath)


class OutputFile:
    def __init__(
        self, outputDirectory: InputDirectory, relativePath, index_col=None, dtype=None
    ):
        self.filePath = outputDirectory.append(relativePath)
        self.index_col = index_col
        self.dtype = dtype
        self.__file = None

    @property
    def file(self):
        if self.__file is None:
            print("Reading file from {0}".format(self.filePath))
            self.__file = pd.read_csv(
                self.filePath, index_col=self.index_col, dtype=None
            )
        return self.__file

    def isDefined(self):
        return self.__file is not None

    def clean(self):
        self.__file = None


class EventsFile(OutputFile):
    def __init__(self, outputDirectory: InputDirectory, iteration: int):
        dtypes = {"type": str, "numPassengers": "Int64"}  # Might come in handy later
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.events.csv.gz".format(iteration),
        ]
        super().__init__(outputDirectory, relativePath, dtype=dtypes)


class LinkStatsFile(OutputFile):
    def __init__(self, outputDirectory: InputDirectory, iteration: int):
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.linkstats.csv.gz".format(iteration),
        ]
        super().__init__(outputDirectory, relativePath)


class InputPlansFile(OutputFile):
    def __init__(self, outputDirectory: InputDirectory):
        relativePath = "plans.csv.gz"
        super().__init__(outputDirectory, relativePath)


class BeamRunInputDirectory(InputDirectory):
    def __init__(self, baseFolderName: str, numberOfIterations: int = 0):
        super().__init__(baseFolderName)
        self.eventsFile = EventsFile(self, numberOfIterations)
        self.inputPlansFile = InputPlansFile(self)
        self.linkStatsFile = LinkStatsFile(self, numberOfIterations)
