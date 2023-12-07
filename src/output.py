import os

import pandas as pd
import hashlib

from src.input import (
    BeamRunInputDirectory,
    InputDirectory,
    PersonsFile,
    ActivitySimRunInputDirectory,
)
from src.transformations import fixPathTraversals, getLinkStats, filterPersons


class OutputDataDirectory:
    """
    Represents an output data directory where results of postprocessing will be saved.

    Attributes:
        path (str): The path to the output data directory.
    """

    def __init__(self, path):
        self.path = path


class BeamOutputData:
    """
    Represents output data related to a Beam run.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        beamRunInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        personEntersVehicleEvents (PersonEntersVehicleEvents): Person enters vehicle events data.
        modeChoiceEvents (ModeChoiceEvents): Mode choice events data.
        modeVMT (ModeVMT): Mode vehicle miles traveled data.
        linkStatsFromPathTraversals (LinkStatsFromPathTraversals): Alternative linkstats
    """

    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamRunInputDirectory: BeamRunInputDirectory,
    ):
        """
        Initializes a BeamOutputData instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamRunInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        """
        self.outputDataDirectory = outputDataDirectory
        self.beamRunInputDirectory = beamRunInputDirectory

        self.beamRunInputDirectory.eventsFile.collectEvents(
            ["PathTraversal", "PersonEntersVehicle", "ModeChoice"]
        )

        self.pathTraversalEvents = PathTraversalEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.personEntersVehicleEvents = PersonEntersVehicleEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.modeChoiceEvents = ModeChoiceEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.modeVMT = ModeVMT(self.outputDataDirectory, self.pathTraversalEvents)
        self.linkStatsFromPathTraversals = LinkStatsFromPathTraversals(
            self.outputDataDirectory, self.pathTraversalEvents
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
        self, outputDataDirectory: OutputDataDirectory, inputDirectory: InputDirectory
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
            self.outputDataDirectory.path,
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
                print(self._dataFrame.dtypes)
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
        self.dataFrame.to_csv(
            os.path.join(self.outputDataDirectory.path, name + ".csv")
        )


class PathTraversalEvents(OutputDataFrame):
    """
    Represents path traversal events data extracted from an events file and then preprocessed

    Attributes:
        beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
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
            df = self.beamInputDirectory.eventsFile.file
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
        outputDataDirectory: OutputDataDirectory,
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
            df = self.beamInputDirectory.eventsFile.file
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
        outputDataDirectory: OutputDataDirectory,
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, beamInputDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def load(self):
        if "ModeChoice" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["ModeChoice"]
        else:
            df = self.beamInputDirectory.eventsFile.file
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
        outputDataDirectory: OutputDataDirectory,
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
        outputDataDirectory: OutputDataDirectory,
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
        outputDataDirectory: OutputDataDirectory,
        activitySimOutputData: ActivitySimRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, activitySimOutputData)
        self.activitySimOutputData = activitySimOutputData
        self.indexedOn = "person_id"

    def preprocess(self, df):
        return filterPersons(df)

    def load(self):
        return self.activitySimOutputData.personsFile.file


class MandatoryLocationsByTaz(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        personsFile: ProcessedPersonsFile,
    ):
        super().__init__(outputDataDirectory, personsFile.inputDirectory)
        self.indexedOn = "TAZ"
        self.personsFile = personsFile

    def load(self):
        population = self.personsFile.file["TAZ"].value_counts()
        workplaces = self.personsFile.file["work_zone_id"].value_counts()
        workplaces = workplaces.loc[workplaces.index > 0]
        workplaces.index.set_names("TAZ", inplace=True)
        return pd.concat({"population": population, "jobs": workplaces}, axis=1).fillna(
            0.0
        )


class ActivitySimOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        activitySimRunInputDirectory: ActivitySimRunInputDirectory,
    ):
        self.outputDataDirectory = outputDataDirectory
        self.activitySimRunInputDirectory = activitySimRunInputDirectory

        self.persons = ProcessedPersonsFile(
            self.outputDataDirectory, self.activitySimRunInputDirectory
        )

        self.mandatoryLocationsByTaz = MandatoryLocationsByTaz(
            self.outputDataDirectory, self.persons
        )
