import os

import pandas as pd


class InputDirectory:
    """
    Represents an input directory of raw data for postprocessing

    Attributes:
        directoryPath (str): The path to the directory.
        isLink (bool): Indicates whether the directory path includes a link either to a website or an s3 or gcloud bucket
    """
    def __init__(self, path: str):
        self.directoryPath = path
        self.isLink = "://" in path

    def append(self, relativePath):
        """
        Appends a relative path to the directory path and returns the combined path.

        Parameters:
            relativePath: The relative path to append.

        Returns:
            str: The combined path.
        """
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
    """
    Represents a raw output file. It can be given additional optional properties like index_col and dtype.

    Attributes:
        filePath (str): The path to the output file.
        index_col: Optional parameter for specifying the column to use as the row labels.
        dtype: Optional parameter for specifying column data types.
        __file: Internal variable to store the loaded file.
    """
    def __init__(
        self, outputDirectory: InputDirectory, relativePath, index_col=None, dtype=None
    ):
        self.filePath = outputDirectory.append(relativePath)
        self.index_col = index_col
        self.dtype = dtype
        self.__file = None

    @property
    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self.__file is None:
            print("Reading file from {0}".format(self.filePath))
            self.__file = pd.read_csv(
                self.filePath, index_col=self.index_col, dtype=None
            )
        return self.__file

    def isDefined(self):
        """
        Checks if the file is defined (loaded).

        Returns:
            bool: True if the file is defined, False otherwise.
        """
        return self.__file is not None

    def clean(self):
        """
        Resets the internal file variable, allowing for reloading the file or clearing memory.
        """
        self.__file = None


class EventsFile(OutputFile):
    """
    Represents an events file produced by BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """
    def __init__(self, inputDirectory: InputDirectory, iteration: int):
        """
        Initializes an EventsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The input directory where the file is stored.
            iteration (int): The BEAM iteration number to use.
        """
        dtypes = {"type": str, "numPassengers": "Int64"}  # Might come in handy later
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.events.csv.gz".format(iteration),
        ]
        super().__init__(inputDirectory, relativePath, dtype=dtypes)


class LinkStatsFile(OutputFile):
    """
    Represents a linkStats file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """
    def __init__(self, outputDirectory: InputDirectory, iteration: int):
        """
        Initializes a LinkStatsFile instance.

        Parameters:
            outputDirectory (InputDirectory): The output directory where the file will be stored.
            iteration (int): The iteration number.
        """
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.linkstats.csv.gz".format(iteration),
        ]
        super().__init__(outputDirectory, relativePath)


class InputPlansFile(OutputFile):
    """
    Represents an input plans file generated for BEAM by ActivitySim.

    Attributes:
        (inherits attributes from OutputFile)
    """
    def __init__(self, outputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            outputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "plans.csv.gz"
        super().__init__(outputDirectory, relativePath)


class BeamRunInputDirectory(InputDirectory):
    """
    Represents an input directory specific to a BEAM run.

    Attributes:
        eventsFile (EventsFile): The events file for the Beam run.
        inputPlansFile (InputPlansFile): The input plans file for the Beam run.
        linkStatsFile (LinkStatsFile): The link stats file for the Beam run.
        (inherits attributes from InputDirectory)
    """
    def __init__(self, baseFolderName: str, numberOfIterations: int = 0):
        """
        Initializes a BeamRunInputDirectory instance.

        Parameters:
            baseFolderName (str): The base folder name for the Beam run.
            numberOfIterations (int): The number of iterations for the Beam run.
        """
        super().__init__(baseFolderName)
        self.eventsFile = EventsFile(self, numberOfIterations)
        self.inputPlansFile = InputPlansFile(self)
        self.linkStatsFile = LinkStatsFile(self, numberOfIterations)
