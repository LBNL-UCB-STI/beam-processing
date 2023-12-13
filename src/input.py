import os
from typing import Iterable

import pandas as pd
import numpy as np
import geopandas as gpd
import openmatrix as omx
import urllib.request


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


class RawOutputFile:
    """
    Represents a raw output file. It can be given additional optional properties like index_col and dtype.

    Attributes:
        filePath (str): The path to the output file.
        index_col: Optional parameter for specifying the column to use as the row labels.
        dtype: Optional parameter for specifying column data types.
        __file: Internal variable to store the loaded file.
    """

    def __init__(
        self,
        outputDirectory: InputDirectory,
        relativePath,
        index_col=None,
        dtype=None,
        file=None,
    ):
        self.filePath = outputDirectory.append(relativePath)
        self.outputDirectory = outputDirectory
        self.index_col = index_col
        self.dtype = dtype
        self.__file = file

    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self.__file is None:
            print("Reading file from {0}".format(self.filePath))
            try:
                self.__file = pd.read_csv(
                    self.filePath, index_col=self.index_col, dtype=None
                )
            except FileNotFoundError:
                print("File at {0} does not exist".format(self.filePath))
                return None
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


class EventsFile(RawOutputFile):
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
        dtypes = {
            "type": str,
            "numPassengers": "Int64",
            "driver": "str",
            "riders": "str",
            "linkTravelTime": "str",
            "links": "str",
        }
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.events.csv.gz".format(iteration),
        ]
        super().__init__(inputDirectory, relativePath, dtype=dtypes)
        self.eventTypes = dict()
        self.__chunksize = 1000000

    def collectEvents(self, eventTypes: list):
        """
        Collects specific event types from the raw events file and stores them in the EventsFile instance.

        Parameters:
            eventTypes (list): A list of event types to collect from the raw events file.
        """
        __listOfFrames = {eventType: [] for eventType in eventTypes}
        for chunk in pd.read_csv(
            self.filePath,
            chunksize=self.__chunksize,
            dtype={
                "driver": "str",
                "riders": "str",
                "linkTravelTime": "str",
                "links": "str",
            },
        ):
            for eventType in eventTypes:
                __listOfFrames[eventType].append(
                    chunk.loc[chunk["type"] == eventType, :].dropna(axis=1, how="all")
                )
        for eventType in eventTypes:
            print("Extracting {0} events from raw events file".format(eventType))
            self.eventTypes[eventType] = pd.concat(
                __listOfFrames.pop(eventType), axis=0
            )


class LinkStatsFile(RawOutputFile):
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
        super().__init__(outputDirectory, relativePath, index_col=["link", "hour"])


class NetworkFile(RawOutputFile):
    """
    Represents a network file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, outputDirectory: InputDirectory):
        """
        Initializes a Network instance.

        Parameters:
            outputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "network.csv.gz"
        super().__init__(outputDirectory, relativePath, index_col="linkId")


class InputPlansFile(RawOutputFile):
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


class PersonsFile(RawOutputFile):
    def __init__(self, outputDirectory: InputDirectory):
        relativePath = "persons.csv.gz"
        super().__init__(outputDirectory, relativePath, index_col="person_id")


class HouseholdsFile(RawOutputFile):
    def __init__(self, outputDirectory: InputDirectory):
        relativePath = "households.csv.gz"
        super().__init__(outputDirectory, relativePath, index_col="household_id")


class TripsFile(RawOutputFile):
    def __init__(self, outputDirectory: InputDirectory):
        relativePath = "final_trips.csv.gz"
        super().__init__(
            outputDirectory,
            relativePath,
            index_col="trip_id",
            dtype={"household_id": int, "person_id": int, "tour_id": int},
        )


class ToursFile(RawOutputFile):
    def __init__(self, outputDirectory: InputDirectory):
        relativePath = "final_tours.csv.gz"
        super().__init__(
            outputDirectory,
            relativePath,
            index_col="tour_id",
            dtype={"household_id": int, "person_id": int, "trip_id": int},
        )


class SkimsFile(RawOutputFile):
    """
    Represents a skims file used in activity-based models.

    Attributes:
        (inherits attributes from RawOutputFile)
    """

    def __init__(self, outputDirectory: InputDirectory):
        """
        Initializes a SkimsFile instance.

        Parameters:
            outputDirectory (InputDirectory): The output directory where the file is stored.
        """
        relativePath = ["activitysim", "data", "data", "skims.omx"]
        # TODO: Support local files too
        loc = ".tmp/skims.omx"
        if not os.path.exists(loc):
            url = outputDirectory.append(relativePath)
            urllib.request.urlretrieve(url, ".tmp/skims.omx")
        sk = omx.open_file(loc, "r")
        distMat = np.array(sk["SOV_DIST__AM"])
        transitTimeMat = np.array(sk["WLK_TRN_WLK_IVT__AM"])
        distDf = (
            pd.DataFrame(
                distMat,
                index=pd.Index(np.arange(1, 1455), name="Origin"),
                columns=pd.Index(np.arange(1, 1455), name="Destination"),
            )
            .stack()
            .rename("DistanceMiles")
        ).to_frame()
        distDf["transitTravelTimeHours"] = pd.DataFrame(
            transitTimeMat / 100.0 / 60.0,
            index=pd.Index(np.arange(1, 1455), name="Origin"),
            columns=pd.Index(np.arange(1, 1455), name="Destination"),
        ).stack()
        super().__init__(outputDirectory, loc, file=distDf)
        sk.close()


class ActivitySimRunInputDirectory(InputDirectory):
    def __init__(self, baseFolderName: str):
        super().__init__(baseFolderName)
        self.householdsFile = HouseholdsFile(self)
        self.personsFile = PersonsFile(self)
        self.tripsFile = TripsFile(self)
        self.toursFile = ToursFile(self)


class PilatesRunInputDirectory(InputDirectory):
    def __init__(
        self,
        baseFolderName: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations=0,
    ):
        super().__init__(baseFolderName)
        self.asimRuns = dict()
        self.beamRuns = dict()
        self.skims = SkimsFile(self)
        for year in years:
            for asimLiteIteration in [-1, *np.arange(asimLiteIterations) + 1]:
                relPath = [
                    "activitysim",
                    "year-{0}-iteration-{1}".format(year, asimLiteIteration),
                ]
                print("Loading year {0} it {1}".format(year, asimLiteIteration))
                self.asimRuns[(year, asimLiteIteration)] = ActivitySimRunInputDirectory(
                    self.append(relPath)
                )
                relPath = [
                    "beam",
                    "year-{0}-iteration-{1}".format(year, asimLiteIteration),
                ]
                self.beamRuns[(year, asimLiteIteration)] = BeamRunInputDirectory(
                    self.append(relPath), beamIterations
                )


class Geometry:
    def __init__(self):
        self.region = None
        self.crs = None
        self._gdf = None
        self.unit = None
        self._path = None
        self._inputcrs = None
        self._gdf = None

    @property
    def gdf(self):
        if self._gdf is None:
            self.load()
        return self._gdf

    def load(self):
        self._gdf = gpd.read_file(self._path)

    def zoneToCountyMap(self):
        return NotImplementedError("This region is not defined yet")


class SfBayGeometry(Geometry):
    def __init__(self):
        super().__init__()
        self.region = "SFBay"
        self.crs = "epsg:26910"
        self.unit = "TAZ"
        self._path = "geoms/sfbay-tazs-epsg-26910.shp"

        self.load()

    def zoneToCountyMap(self):
        return self._gdf["county"].to_dict()
