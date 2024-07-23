import hashlib
import os
import pathlib
import subprocess
from gzip import BadGzipFile
from io import BytesIO, StringIO
from typing import Iterable, Optional, Dict
from urllib.error import HTTPError, URLError
from zipfile import ZipFile
import requests
from google.cloud import storage


from tables import HDF5ExtError
from tqdm import tqdm
import shutil

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
        _file: Internal variable to store the loaded file.
    """

    def __init__(
        self,
        inputDirectory: InputDirectory,
        relativePath,
        index_col=None,
        dtype=None,
        file=None,
    ):
        self.filePath = inputDirectory.append(relativePath)
        self.inputDirectory = inputDirectory
        self.index_col = index_col
        self.dtype = dtype
        self._file = file

    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._file is None:
            print("Reading file from {0}".format(self.filePath))
            try:
                self._file = pd.read_csv(
                    self.filePath, index_col=self.index_col, dtype=None
                )
                if self._file.columns[0].startswith("<!"):
                    # Catch reading an html file as a table
                    raise pd.errors.ParserError
            except (FileNotFoundError, HTTPError):
                print("File at {0} does not exist".format(self.filePath))
                return None
            except (BadGzipFile, pd.errors.ParserError) as e:
                print("Initial download failed")
                bucket = storage.Client().get_bucket(self.filePath.split("/")[3])
                blob = bucket.get_blob("/".join(self.filePath.split("/")[4:]))
                if blob is not None:
                    if self.filePath.endswith("gz"):
                        fmt = ".csv.gz"
                    else:
                        fmt = ".csv"
                    path = pathlib.Path.cwd().joinpath(".tmp", self.hash() + fmt)
                    print("Downloading file from gcloud to {0}".format(path))
                    blob.download_to_filename(path)

                    self._file = pd.read_csv(path, index_col=self.index_col, dtype=None)
                    print("Success! Deleting temporary file")
                    os.remove(path)
                else:
                    print("Giving up!")
                    return None
        return self._file

    def hash(self):
        """
        Generates a hash based on the input and class name. Also include whether we've generated it from linkstats or events

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        for s in (self.inputDirectory.directoryPath, self.__class__.__name__):
            m.update(s.encode())
        return m.hexdigest()

    def isDefined(self):
        """
        Checks if the file is defined (loaded).

        Returns:
            bool: True if the file is defined, False otherwise.
        """
        return self._file is not None

    def clean(self):
        """
        Resets the internal file variable, allowing for reloading the file or clearing memory.
        """
        self._file = None


class Geometry:
    def __init__(self):
        self.region = None
        self.crs = None
        self._gdf = None
        self.unit = None
        self._path = None
        self._inputcrs = None
        self._gdf = None
        self.index = None
        self._otherFiles = dict()

    @property
    def gdf(self):
        if self._gdf is None:
            self.load()
        return self._gdf

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )

    def zoneToCountyMap(self):
        return NotImplementedError("This region is not defined yet")


class SfBayGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "SFBay"
        self.crs = "epsg:26910"
        self.unit = "TAZ"
        self.index = "taz1454"
        self._path = "geoms/sfbay-tazs-epsg-26910.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        return self._gdf.set_index(self.index)["areatype10"].to_dict()


class AustinGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Austin"
        self.crs = "epsg:26910"
        self.unit = "BG"
        self.index = "TAZ"
        self._path = "geoms/block_group_austin_26910.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Austin")


class SeattleGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Seattle"
        self.crs = "epsg:32048"
        self.unit = "BG"
        self.index = "OBJECTID"
        self._path = "geoms/block-groups-32048.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Seattle")

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )
        self._gdf.rename(columns={"county_nam": "county"}, inplace=True)


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
            "person": "str",
            "vehicle": "str",
            "parkingTaz": "str",
        }
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.events.csv.gz".format(iteration),
        ]
        super().__init__(inputDirectory, relativePath, dtype=dtypes)
        self.eventTypes = dict()
        self.__chunksize = 5000000

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
                "person": "str",
                "vehicle": "str",
                "parkingTaz": "str",
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

    def clearEvents(self):
        print("Clearing events memory")
        self.eventTypes.clear()


class LinkStatsFile(RawOutputFile):
    """
    Represents a linkStats file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory, iteration: int):
        """
        Initializes a LinkStatsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
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

    def __init__(self, inputDirectory: InputDirectory, geometry: Geometry):
        """
        Initializes a Network instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
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

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "plans.csv.gz"
        super().__init__(inputDirectory, relativePath)


class ReplanningEventReasonFile(RawOutputFile):
    """
    Represents all replanning events in BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "replanningEventReason.csv"
        super().__init__(inputDirectory, relativePath)


class ScoreStatsFile(RawOutputFile):
    """
    Keeps track of agent scores in a BEAM run

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
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


class BeamRunInputDirectory(InputDirectory):
    """
    Represents an input directory specific to a BEAM run.

    Attributes:
        eventsFile (EventsFile): The events file for the Beam run.
        inputPlansFile (InputPlansFile): The input plans file for the Beam run.
        linkStatsFile (LinkStatsFile): The link stats file for the Beam run.
        (inherits attributes from InputDirectory)
    """

    def __init__(
        self,
        baseFolderName: str,
        numberOfIterations: int = 0,
        geometry: Optional[Geometry] = None,
        region: Optional[str] = None,
    ):
        """
        Initializes a BeamRunInputDirectory instance.

        Parameters:
            baseFolderName (str): The base folder name for the Beam run.
            numberOfIterations (int): The number of iterations for the Beam run.
        """
        super().__init__(baseFolderName)
        self.eventsFile = EventsFile(self, numberOfIterations)
        self.numberOfIterations = numberOfIterations
        self.inputPlansFile = InputPlansFile(self)
        self.replanningEventReasonFile = ReplanningEventReasonFile(self)
        self.scoreStatsFile = ScoreStatsFile(self)
        self.__linkStatsFile = {
            numberOfIterations: LinkStatsFile(self, numberOfIterations)
        }
        if (region is not None) & (geometry is None):
            if region == "SFBay":
                self.geometry = SfBayGeometry(
                    otherFiles={
                        "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
                    }
                )
            elif region == "Austin":
                self.geometry = AustinGeometry(otherFiles=dict())
            elif region == "Seattle":
                self.geometry = SeattleGeometry(otherFiles=dict())
            else:
                self.geometry = Geometry()
        else:
            self.geometry = geometry
        self.networkFile = NetworkFile(self, self.geometry)

    def linkStatsFile(self, numberOfIterations: Optional[int] = None):
        if numberOfIterations is None:
            numberOfIterations = self.numberOfIterations
        if numberOfIterations not in self.__linkStatsFile:
            self.__linkStatsFile[numberOfIterations] = LinkStatsFile(
                self, numberOfIterations
            )
        return self.__linkStatsFile[numberOfIterations]


class TripUtilitiesFiles(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "trip_mode_choice.zip"
        super().__init__(inputDirectory, relativePath, index_col="person_id")
        # self._file = None

    def __hash(self):
        """
        Generates a hash based on the input and class name. Also include whether we've generated it from linkstats or events

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        for s in (self.inputDirectory.directoryPath, self.__class__.__name__):
            m.update(s.encode())
        return m.hexdigest()

    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._file is None:
            out = dict()
            folderName = os.path.join(".tmp", self.__hash())
            if not os.path.exists(folderName):
                os.makedirs(folderName)
                print("Reading files from {0}".format(self.filePath))
                try:
                    with urllib.request.urlopen(self.filePath) as zipresp:
                        with ZipFile(BytesIO(zipresp.read())) as zfile:
                            for ls in tqdm(zfile.filelist):
                                if ls.filename.endswith("utilities.csv"):
                                    if not os.path.exists(
                                        os.path.join(folderName, ls.filename)
                                    ):
                                        zfile.extract(ls.filename, folderName)
                                    groupName = ls.filename.split("/")[1].split("_")[0]
                                    df = pd.read_csv(
                                        os.path.join(folderName, ls.filename),
                                        index_col="trip_id",
                                    )
                                    out[groupName] = df
                    self._file = pd.concat(out, names=["division", "trip_id"])
                except FileNotFoundError:
                    print("File at {0} does not exist".format(self.filePath))
                    return None
            else:
                files = os.listdir(os.path.join(folderName, "trip_mode_choice"))
                print("Reading saved utility files")
                for file in tqdm(files):
                    if file.endswith("utilities.csv"):
                        groupName = file.split("_")[0]
                        df = pd.read_csv(
                            os.path.join(folderName, "trip_mode_choice", file),
                            index_col="trip_id",
                        )
                        out[groupName] = df
                self._file = pd.concat(out, names=["division", "trip_id"])
        return self._file

    def getInitialDivisionMapping(self):
        trip_id_to_division = (
            self.file()
            .index.to_frame()
            .reset_index("division", drop=True)["division"]
            .to_dict()
        )
        print(
            "Returning trip to division mapping for {0} trips".format(
                len(trip_id_to_division)
            )
        )
        return trip_id_to_division

    def split(self, trip_id_to_division) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        utils = self.file().reset_index()
        utils["division_fixed"] = utils["division"].astype(int).map(trip_id_to_division)
        gb = utils.groupby("division_fixed")
        division_to_utils = {a: b for a, b in gb}
        return division_to_utils, trip_id_to_division


class PersonsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "persons.csv.gz"
        super().__init__(inputDirectory, relativePath, index_col="person_id")

    def split(self, person_id_to_division) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        households = self.file()["household_id"].reset_index()
        households["division"] = households["person_id"].map(person_id_to_division)
        household_id_to_division = households.set_index("person_id")[
            "division"
        ].to_dict()
        gb = self.file().groupby(person_id_to_division)
        division_to_persons = {f: d for f, d in gb}
        return division_to_persons, household_id_to_division


class HouseholdsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "households.csv.gz"
        super().__init__(inputDirectory, relativePath, index_col="household_id")

    def split(
        self, household_id_to_division
    ) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        gb = self.file().groupby(household_id_to_division)
        division_to_households = {f: d for f, d in gb}
        return division_to_households, dict()


class TripsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "final_trips.csv.gz"
        super().__init__(
            inputDirectory,
            relativePath,
            index_col="trip_id",
            dtype={"household_id": int, "person_id": int, "tour_id": int},
        )

    def split(self, trip_id_to_division) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        persons = self.file()["person_id"].reset_index()
        persons["division"] = persons["trip_id"].map(trip_id_to_division)

        divToPersons = persons.groupby("division").agg({"person_id": "unique"})
        tempMap = dict()
        tempMap[divToPersons.iloc[0].name] = set(divToPersons.iloc[0].person_id)
        for div, ps in divToPersons.iterrows():
            setPersons = set(ps.person_id)
            unassigned = True
            for finalDiv, finalPersons in tempMap.items():
                if not setPersons.isdisjoint(finalPersons):
                    finalPersons.update(setPersons)
                    unassigned = False
            if unassigned:
                tempMap[div] = setPersons

        outputMap = dict()
        outputMap[divToPersons.iloc[0].name] = tempMap[divToPersons.iloc[0].name]
        for div, setPersons in tempMap.items():
            unassigned = True
            for finalDiv, finalPersons in outputMap.items():
                if not setPersons.isdisjoint(finalPersons):
                    finalPersons.update(setPersons)
                    unassigned = False
            if unassigned:
                outputMap[div] = setPersons

        person_id_to_division = (
            pd.concat(
                [
                    pd.MultiIndex.from_product(
                        [[a], list(b)], names=["division", "person_id"]
                    ).to_frame(False)
                    for a, b in outputMap.items()
                ]
            )
            .set_index("person_id")
            .to_dict()["division"]
        )

        trips = self.file()
        trips["division_new"] = trips["person_id"].map(person_id_to_division)
        gb = trips.groupby("division_new")
        division_to_trips = {f: d for f, d in gb}
        trip_id_to_division_new = trips["division_new"].to_dict()
        return division_to_trips, person_id_to_division, trip_id_to_division_new


class ToursFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "final_tours.csv.gz"
        super().__init__(
            inputDirectory,
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

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes a SkimsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file is stored.
        """

        # TODO: Support local files too
        loc = ".tmp/" + self.hash(inputDirectory) + ".omx"
        if False:#not os.path.exists(loc):
            try:
                relativePath = ["activitysim", "data", "data", "skims.omx"]
                url = inputDirectory.append(relativePath)
                urllib.request.urlretrieve(url, loc)
            except:
                try:
                    relativePath = ["activitysim", "data", "skims.omx"]
                    url = inputDirectory.append(relativePath)
                    urllib.request.urlretrieve(url, loc)
                except:
                    print('ops')
                    # relativePath = ["activitysim", "data", "skims.omx"]
                    # url = os.path.join(inputDirectory.directoryPath.replace("gs://","https://storage.googleapis.com/"), *relativePath)
                    # urllib.request.urlretrieve(url, loc)
        else:
            print('loading skims from ', loc)
        sk = omx.open_file(loc, "r")
        distMat = np.array(sk["SOV_DIST__AM"])

        transitModes = ["COM", "LOC", "HVY", "LRF"]
        distDf = (
            pd.DataFrame(
                distMat,
                index=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Origin"),
                columns=pd.Index(
                    np.arange(1, distMat.shape[0] + 1), name="Destination"
                ),
            )
            .stack()
            .rename("DistanceMiles")
        ).to_frame()
        for m in transitModes:
            transitTimeMat = np.array(sk["WLK_{0}_WLK_TOTIVT__AM".format(m)])
            distDf["transitTravelTimeHours_" + m] = pd.DataFrame(
                transitTimeMat / 100.0 / 60.0,
                index=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Origin"),
                columns=pd.Index(
                    np.arange(1, distMat.shape[0] + 1), name="Destination"
                ),
            ).stack()
            transitWaitMat = (
                np.array(sk["WLK_{0}_WLK_IWAIT__AM".format(m)])
                + np.array(sk["WLK_{0}_WLK_XWAIT__AM".format(m)])
                + np.array(sk["WLK_{0}_WLK_WACC__AM".format(m)])
                + np.array(sk["WLK_{0}_WLK_WEGR__AM".format(m)])
                + np.array(sk["WLK_{0}_WLK_WAUX__AM".format(m)])
            )
            distDf["transitWaitTimeHours_" + m] = pd.DataFrame(
                transitWaitMat / 100.0 / 60.0,
                index=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Origin"),
                columns=pd.Index(
                    np.arange(1, distMat.shape[0] + 1), name="Destination"
                ),
            ).stack()
        driveTimeMat = np.array(sk["SOV_TIME__AM"])
        distDf["driveTimeHours"] = pd.DataFrame(
            driveTimeMat / 100.0 / 60.0,
            index=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Origin"),
            columns=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Destination"),
        ).stack()
        walkTimeMat = distMat / 2.5
        distDf["walkTimeHours"] = pd.DataFrame(
            walkTimeMat / 100.0 / 60.0,
            index=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Origin"),
            columns=pd.Index(np.arange(1, distMat.shape[0] + 1), name="Destination"),
        ).stack()
        super().__init__(inputDirectory, loc, file=distDf)
        sk.close()

    def hash(self, inputDirectory=None):
        if inputDirectory is None:
            inputDirectory = self.inputDirectory
        m = hashlib.md5()
        for s in (
            inputDirectory.directoryPath,
            self.__class__.__name__,
        ):
            m.update(s.encode())
        return m.hexdigest()


class ActivitySimRunInputDirectory(InputDirectory):
    def __init__(self, baseFolderName: str, geometry=Geometry()):
        super().__init__(baseFolderName)
        self.householdsFile = HouseholdsFile(self)
        self.personsFile = PersonsFile(self)
        self.tripsFile = TripsFile(self)
        self.toursFile = ToursFile(self)
        self.tripUtilitiesFiles = TripUtilitiesFiles(self)
        self.geometry = geometry

    def getSplitData(self):
        trip_id_to_division_raw = self.tripUtilitiesFiles.getInitialDivisionMapping()
        (
            division_to_trips,
            person_id_to_division,
            trip_id_to_division,
        ) = self.tripsFile.split(trip_id_to_division_raw)
        division_to_utilities, _ = self.tripUtilitiesFiles.split(trip_id_to_division)
        division_to_persons, household_id_to_division = self.personsFile.split(
            person_id_to_division
        )
        division_to_households, _ = self.householdsFile.split(household_id_to_division)
        return (
            division_to_utilities,
            division_to_trips,
            division_to_persons,
            division_to_households,
            person_id_to_division,
        )


class PilatesRunInputDirectory(InputDirectory):
    def __init__(
        self,
        baseFolderName: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations=0,
        region="SFBay",
    ):
        super().__init__(baseFolderName)
        self.asimRuns = dict()
        self.beamRuns = dict()
        try:
            self.skims = SkimsFile(self)
        except Exception as e:
            print("Skipping skims")
        if region == "SFBay":
            self.geometry = SfBayGeometry(
                otherFiles={
                    "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
                }
            )
        elif region == "Seattle":
            self.geometry = SeattleGeometry(otherFiles=dict())
        elif region == "Austin":
            self.geometry = AustinGeometry(otherFiles=dict())
        else:
            self.geometry = Geometry()
        for year in years:
            for asimLiteIteration in [-1, *np.arange(asimLiteIterations) + 1]:
                relPath = ["activitysim"]
                if not self.isLink:
                    relPath.append("output")
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                print("Loading year {0} it {1}".format(year, asimLiteIteration))
                self.asimRuns[(year, asimLiteIteration)] = ActivitySimRunInputDirectory(
                    self.append(relPath), self.geometry
                )
                relPath = ["beam"]
                if not self.isLink:
                    relPath.append("beam_output")
                    relPath.append(region.lower())
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                self.beamRuns[(year, asimLiteIteration)] = BeamRunInputDirectory(
                    self.append(relPath), beamIterations, self.geometry
                )
