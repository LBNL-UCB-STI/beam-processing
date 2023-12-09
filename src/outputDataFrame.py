import hashlib
import os
from typing import Dict, Tuple

import pandas as pd

from src.input import (
    InputDirectory,
    BeamRunInputDirectory,
    ActivitySimRunInputDirectory,
    PilatesRunInputDirectory,
)
from src.transformations import (
    fixPathTraversals,
    getLinkStats,
    filterPersons,
    filterHouseholds,
    filterTrips,
)


class OutputDataFrame:
    """
    Represents an output DataFrame with basic functionality like loading and preprocessing.
    Designed to load in and preprocess raw BEAM outputs stored in an OutputDataDirectory

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        _dataFrame (pd.DataFrame): Internal variable to store the loaded DataFrame.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", inputDirectory: InputDirectory
    ):
        """
        Initializes an OutputDataFrame instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
        """
        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        self._dataFrame = None
        self._diskLocation = os.path.join(
            ".tmp",
            self.hash() + ".parquet",
        )
        self.indexedOn = None

    def hash(self):
        m = hashlib.md5()
        for s in (self.inputDirectory.directoryPath, self.__class__.__name__):
            m.update(s.encode())
        return m.hexdigest()

    @property
    def cached(self) -> bool:
        return os.path.exists(self._diskLocation)

    def clearCache(self):
        if self.cached:
            os.remove(self._diskLocation)
        self._dataFrame = None

    def load(self):
        """
        Abstract method for loading data into a DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        pass

    @property
    def dataFrame(self) -> pd.DataFrame:
        """
        Property to lazily load the DataFrame and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._dataFrame is None:
            if not self.cached:
                df = self.load()
                self._dataFrame = self.preprocess(df)
                self._dataFrame.to_parquet(self._diskLocation)
            else:
                print(
                    "Reading {0} file from {1}".format(
                        self.__class__.__name__, self._diskLocation
                    )
                )
                self._dataFrame = pd.read_parquet(self._diskLocation)
        return self._dataFrame

    @dataFrame.setter
    def dataFrame(self, df: pd.DataFrame):
        """
        Setter for the dataFrame property.

        Parameters:
            df (pd.DataFrame): The DataFrame to set.
        """
        self._dataFrame = df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for preprocessing the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        return df

    def toCsv(self):
        """
        Saves the DataFrame to a CSV file in the output data directory.
        """
        name = self.__class__.__name__
        if not os.path.exists(self.outputDataDirectory.path):
            os.makedirs(self.outputDataDirectory.path)
        self.dataFrame.to_csv(
            os.path.join(self.outputDataDirectory.path, name + ".csv")
        )

    def addMapping(self, mapping: dict, fromCol: str, toCol: str):
        self.dataFrame[toCol] = self.dataFrame.index.get_level_values(fromCol).map(
            mapping
        )
        return self.dataFrame

    def unstackColumn(self, col, index):
        return self.dataFrame[col].unstack(index)


class PathTraversalEvents(OutputDataFrame):
    """
    Represents path traversal events data extracted from an events file and then preprocessed

    Attributes:
        beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunInputDirectory,
    ):
        """
        Initializes a PathTraversalEvents instance from raw events file

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        """
        super().__init__(outputDataDirectory, beamInputDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def preprocess(self, df):
        """
        Preprocesses the path traversal events DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        return fixPathTraversals(df)

    def load(self):
        """
        Loads path traversal events data from the Beam input directory, dropping unnecessary columns

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if "PathTraversal" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["PathTraversal"]
        else:
            df = self.beamInputDirectory.eventsFile.file()
            df = df.loc[df["type"] == "PathTraversal", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class PersonEntersVehicleEvents(OutputDataFrame):
    """
    Represents person enters vehicle events data from the raw events file.

    Attributes:
        beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunInputDirectory,
    ):
        """
        Initializes a PersonEntersVehicleEvents instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        """
        super().__init__(outputDataDirectory, beamInputDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def load(self):
        """
        Loads person enters vehicle events data from the Beam input directory.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if "PersonEntersVehicle" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["PersonEntersVehicle"]
        else:
            df = self.beamInputDirectory.eventsFile.file()
            df = df.loc[df["type"] == "PersonEntersVehicle", :].dropna(
                axis=1, how="all"
            )
        df.index.name = "event_id"
        return df


class ModeChoiceEvents(OutputDataFrame):
    """
    Represents mode choice events data.

    Attributes:
        beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, beamInputDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def load(self):
        if "ModeChoice" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["ModeChoice"]
        else:
            df = self.beamInputDirectory.eventsFile.file()
            df = df.loc[df["type"] == "ModeChoice", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class ModeVMT(OutputDataFrame):
    """
    Represents mode vehicle miles traveled data, calculated from the PathTraversalEvents

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
    ):
        """
        Initializes a ModeVMT instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        super().__init__(outputDataDirectory, pathTraversalEvents.inputDirectory)
        self.indexedOn = "pathTraversalMode"
        self.pathTraversalEvents = pathTraversalEvents

    def load(self):
        """
        Aggregates mode vehicle miles traveled data from the path traversal events data.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        df = self.pathTraversalEvents.dataFrame.groupby("mode_extended").agg(
            {"vehicleMiles": "sum"}
        )
        df.index.name = self.indexedOn
        return df


class LinkStatsFromPathTraversals(OutputDataFrame):
    """
    Alternative linkstats file, calculated from the PathTraversalEvents

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
    ):
        """
        Initializes a ModeVMT instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        super().__init__(outputDataDirectory, pathTraversalEvents.inputDirectory)
        self.indexedOn = ["linkId", "hour"]
        self.pathTraversalEvents = pathTraversalEvents

    def load(self):
        """
        Aggregates mode vehicle miles traveled data from the path traversal events data.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        df = self.pathTraversalEvents.dataFrame
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return getLinkStats(df)


class ProcessedPersonsFile(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, activitySimOutputData)
        self.activitySimOutputData = activitySimOutputData
        self.indexedOn = "person_id"

    def preprocess(self, df):
        return filterPersons(df)

    def load(self):
        return self.activitySimOutputData.personsFile.file()


class ProcessedHouseholdsFile(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, activitySimOutputData)
        self.activitySimOutputData = activitySimOutputData
        self.indexedOn = "household_id"

    def preprocess(self, df):
        return filterHouseholds(df)

    def load(self):
        return self.activitySimOutputData.personsFile.file()


class ProcessedTripsFile(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, activitySimOutputData)
        self.activitySimOutputData = activitySimOutputData
        self.indexedOn = "trip_id"

    def preprocess(self, df):
        return filterTrips(df)

    def load(self):
        return self.activitySimOutputData.tripsFile.file()


class ProcessedSkimsFile(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesOutputData: PilatesRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, pilatesOutputData)
        self.pilatesOutputData = pilatesOutputData
        self.indexedOn = ["Origin", "Destination"]

    def preprocess(self, df):
        return df

    def load(self):
        return self.pilatesOutputData.skims.file()


class MandatoryLocationsByTaz(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        personsFile: ProcessedPersonsFile,
    ):
        super().__init__(outputDataDirectory, personsFile.inputDirectory)
        self.indexedOn = "TAZ"
        self.personsFile = personsFile

    def load(self):
        population = self.personsFile.dataFrame["TAZ"].value_counts().astype(int)
        workplaces = self.personsFile.dataFrame["work_zone_id"].value_counts()
        workplaces = workplaces.loc[workplaces.index > 0]
        workplaces.index.set_names("TAZ", inplace=True)
        return pd.concat({"population": population, "jobs": workplaces}, axis=1).fillna(
            0
        )


class TripModeCount(OutputDataFrame):
    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", tripsFile: ProcessedTripsFile
    ):
        super().__init__(outputDataDirectory, tripsFile.inputDirectory)
        self.indexedOn = "trip_mode"
        self.tripsFile = tripsFile
        self.indices = ["trip_mode"]

    def load(self):
        mapping = {
            "DRIVEALONEPAY": "SOV",
            "DRIVEALONEFREE": "SOV",
            "SHARED2PAY": "HOV",
            "SHARED2FREE": "HOV",
            "WALK": "WALK",
            "SHARED3PAY": "HOV",
            "SHARED3FREE": "HOV",
            "DRIVE_LOC": "DRIVE_TRANSIT",
            "DRIVE_HVY": "DRIVE_TRANSIT",
            "DRIVE_LRF": "DRIVE_TRANSIT",
            "DRIVE_COM": "DRIVE_TRANSIT",
            "WALK_LOC": "WALK_TRANSIT",
            "WALK_HVY": "WALK_TRANSIT",
            "WALK_LRF": "WALK_TRANSIT",
            "WALK_COM": "WALK_TRANSIT",
            "TAXI": "TNC",
            "TNC_SINGLE": "TNC",
            "TNC_SHARED": "TNC",
        }
        return (
            self.tripsFile.dataFrame.replace({"trip_mode": mapping})
            .value_counts(self.indices, normalize=False)
            .to_frame("count")
        )


class TripPMT(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile.inputDirectory)
        self.indexedOn = "trip_mode"
        self.tripsFile = tripsFile
        self.skimsFile = skimsFile
        self.indices = ["trip_mode"]

    def load(self):
        mapping = {
            "DRIVEALONEPAY": "SOV",
            "DRIVEALONEFREE": "SOV",
            "SHARED2PAY": "HOV",
            "SHARED2FREE": "HOV",
            "WALK": "WALK",
            "SHARED3PAY": "HOV",
            "SHARED3FREE": "HOV",
            "DRIVE_LOC": "DRIVE_TRANSIT",
            "DRIVE_HVY": "DRIVE_TRANSIT",
            "DRIVE_LRF": "DRIVE_TRANSIT",
            "DRIVE_COM": "DRIVE_TRANSIT",
            "WALK_LOC": "WALK_TRANSIT",
            "WALK_HVY": "WALK_TRANSIT",
            "WALK_LRF": "WALK_TRANSIT",
            "WALK_COM": "WALK_TRANSIT",
            "TAXI": "TNC",
            "TNC_SINGLE": "TNC",
            "TNC_SHARED": "TNC",
        }
        self.tripsFile.dataFrame["distanceInMiles"] = (
            self.skimsFile.dataFrame["DistanceMiles"]
            .reindex(
                pd.MultiIndex.from_frame(
                    self.tripsFile.dataFrame[["origin", "destination"]]
                ).values
            )
            .values
        )
        return (
            self.tripsFile.dataFrame.replace({"trip_mode": mapping})
            .groupby(self.indices)
            .agg({"distanceInMiles": "sum"})
        )


class TripModeCountByOrigin(TripModeCount):
    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", tripsFile: ProcessedTripsFile
    ):
        super().__init__(outputDataDirectory, tripsFile)
        self.indices = ["trip_mode", "origin"]


class TripModeCountByPrimaryPurpose(TripModeCount):
    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", tripsFile: ProcessedTripsFile
    ):
        super().__init__(outputDataDirectory, tripsFile)
        self.indices = ["trip_mode", "primary_purpose"]


class TripPMTByOrigin(TripPMT):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile, skimsFile)
        self.indices = ["trip_mode", "origin"]


class TripPMTByPrimaryPurpose(TripPMT):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile, skimsFile)
        self.indices = ["trip_mode", "primary_purpose"]


class MandatoryLocationByTazByYear(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        super().__init__(outputDataDirectory, pilatesRunInputDirectory)
        self.pilatesInputDict = pilatesInputDict
        self.__yearToDataFrame = dict()

    def load(self):
        for (yr, it), data in self.pilatesInputDict.items():
            if yr not in self.__yearToDataFrame:
                self.__yearToDataFrame[yr] = data.mandatoryLocationsByTaz.dataFrame
        return pd.concat(self.__yearToDataFrame, names=["Year", "TAZ"])


class TripModeCountByYear(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        super().__init__(outputDataDirectory, pilatesRunInputDirectory)
        self.pilatesInputDict = pilatesInputDict
        self.__lastIterationPerYear = dict()
        self.__yearToDataFrame = dict()

    def load(self):
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it <= self.__lastIterationPerYear[yr]:
                    continue
            else:
                self.__lastIterationPerYear[yr] = it
        for (yr, it), data in self.pilatesInputDict.items():
            if self.__lastIterationPerYear[yr] == it:
                self.__yearToDataFrame[yr] = data.tripModeCount.dataFrame
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame)
        else:
            return pd.DataFrame


class ModeVMTByYear(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        super().__init__(outputDataDirectory, pilatesRunInputDirectory)
        self.pilatesInputDict = pilatesInputDict
        self.__lastIterationPerYear = dict()
        self.__yearToDataFrame = dict()

    def load(self):
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it <= self.__lastIterationPerYear[yr]:
                    continue
            else:
                self.__lastIterationPerYear[yr] = it
        for (yr, it), data in self.pilatesInputDict.items():
            if self.__lastIterationPerYear[yr] == it:
                self.__yearToDataFrame[yr] = data.modeVMT.dataFrame
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame)
        else:
            return pd.DataFrame()
