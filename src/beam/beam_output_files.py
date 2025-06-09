import os
import pathlib

import pandas as pd
from google.cloud import storage

from src.geometry import Geometry
from src.input_base import RawOutputFile, OutputDirectory


class EventsFile(RawOutputFile):
    """
    Represents an events file produced by BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(
        self, inputDirectory: OutputDirectory, iteration: int, file_format="csv"
    ):
        """
        Initializes an EventsFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The input directory where the file is stored.
            iteration (int): The BEAM iteration number to use.
        """
        dtypes = {
            "type": str,
            "numPassengers": "Int64",
            "driver": "str",
            "riders": "str",
            "linkTravelTime": "str",
            "links": "str",
            "person": "str",
            "vehicle": "str",
            "parkingTaz": "str",
        }
        if file_format == "csv":
            filename = "{0}.events.csv.gz".format(iteration)
        elif file_format == "parquet":
            filename = "{0}.events.parquet".format(iteration)
        else:
            raise ValueError("Unsupported file format: {0}".format(file_format))
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            filename,
        ]
        super().__init__(inputDirectory, relativePath, dtype=dtypes)
        self.eventTypes = dict()
        self.__file_format = file_format
        self.__chunksize = 5000000

    def collectEvents(self, eventTypes: list):
        """
        Collects specific event types from the raw events file and stores them in the EventsFile instance.

        Parameters:
            eventTypes (list): A list of event types to collect from the raw events file.
        """
        __listOfFrames = {eventType: [] for eventType in eventTypes}
        if self.__file_format == "csv":
            for chunk in pd.read_csv(
                self.filePath,
                chunksize=self.__chunksize,
                dtype={
                    "driver": "str",
                    "riders": "str",
                    "linkTravelTime": "str",
                    "links": "str",
                    "person": "str",
                    "vehicle": "str",
                    "parkingTaz": "str",
                },
            ):
                for eventType in eventTypes:
                    __listOfFrames[eventType].append(
                        chunk.loc[chunk["type"] == eventType, :].dropna(
                            axis=1, how="all"
                        )
                    )
        elif self.__file_format == "parquet":
            for eventType in eventTypes:
                __listOfFrames[eventType] = [
                    pd.read_parquet(
                        self.filePath,
                        filters=[("type", "==", eventType)],
                        dtype_backend="pyarrow",
                    ).dropna(how="all", axis=1)
                ]
        for eventType in eventTypes:
            print("Extracting {0} events from raw events file".format(eventType))
            self.eventTypes[eventType] = pd.concat(
                __listOfFrames.pop(eventType), axis=0
            )

    def clearEvents(self):
        print("Clearing events memory")
        self.eventTypes.clear()


class LinkStatsFile(RawOutputFile):
    """
    Represents a linkStats file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory, iteration: int):
        """
        Initializes a LinkStatsFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The output directory where the file will be stored.
            iteration (int): The iteration number.
        """
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.linkstats.csv.gz".format(iteration),
        ]
        super().__init__(inputDirectory, relativePath, index_col=["link", "hour"])
        self.iteration = iteration


class NetworkFile(RawOutputFile):
    """
    Represents a network file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory, geometry: Geometry):
        """
        Initializes a Network instance.

        Parameters:
            inputDirectory (OutputDirectory): The output directory where the file will be stored.
            :param geometry:
        """
        relativePath = "network.csv.gz"
        super().__init__(inputDirectory, relativePath, index_col="linkId")
        self.crs = geometry.crs


class InputPlansFile(RawOutputFile):
    """
    Represents an input plans file generated for BEAM by ActivitySim.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The output directory where the file will be stored.
        """
        relativePath = "plans.csv.gz"
        super().__init__(inputDirectory, relativePath)


class ReplanningEventReasonFile(RawOutputFile):
    """
    Represents all replanning events in BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The output directory where the file will be stored.
        """
        relativePath = "replanningEventReason.csv"
        super().__init__(inputDirectory, relativePath)


class ScoreStatsFile(RawOutputFile):
    """
    Keeps track of agent scores in a BEAM run

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The output directory where the file will be stored.
        """
        relativePath = "scorestats.txt"
        super().__init__(inputDirectory, relativePath)

    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._file is None:
            print("Reading file from {0}".format(self.filePath))
            try:
                self._file = pd.read_table(
                    self.filePath, index_col=self.index_col, dtype=None
                )
                if self._file.columns[0].startswith("<!"):
                    raise pd.errors.ParserError
            except FileNotFoundError:
                print("File at {0} does not exist".format(self.filePath))
                return None
            except pd.errors.ParserError:
                print("Initial download failed")
                bucket = storage.Client().get_bucket(self.filePath.split("/")[3])
                blob = bucket.get_blob("/".join(self.filePath.split("/")[4:]))
                if blob is not None:
                    fmt = ".txt"
                    path = pathlib.Path.cwd().joinpath(".tmp", self.hash() + fmt)
                    print("Downloading file from gcloud to {0}".format(path))
                    blob.download_to_filename(path)

                    self._file = pd.read_table(
                        path, index_col=self.index_col, dtype=None
                    )
                    print("Success! Deleting temporary file")
                    os.remove(path)
                else:
                    print("Giving up!")
                    return None
        return self._file
