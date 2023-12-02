import pandas as pd

from src.input import BeamRunInputDirectory
from src.transformations import fixPathTraversals
import os


class OutputDataDirectory:
    def __init__(self, path):
        self.path = path


class BeamOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamRunInputDirectory: BeamRunInputDirectory,
    ):
        self.outputDataDirectory = outputDataDirectory
        self.beamRunInputDirectory = beamRunInputDirectory
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


class OutputDataFrame:
    def __init__(self, outputDataDirectory: OutputDataDirectory):
        self.outputDataDirectory = outputDataDirectory
        self._dataFrame = None
        self.indexedOn = None

    def load(self):
        pass

    @property
    def dataFrame(self) -> pd.DataFrame:
        if self._dataFrame is None:
            df = self.load()
            self._dataFrame = self.preprocess(df)
        return self._dataFrame

    @dataFrame.setter
    def dataFrame(self, df: pd.DataFrame):
        self._dataFrame = df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def toCsv(self):
        name = self.__class__.__name__
        self.dataFrame.to_csv(
            os.path.join(self.outputDataDirectory.path, name + ".csv")
        )


class PathTraversalEvents(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def preprocess(self, df):
        return fixPathTraversals(df)

    def load(self):
        df = self.beamInputDirectory.eventsFile.file
        df = df.loc[df["type"] == "PathTraversal", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class PersonEntersVehicleEvents(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def load(self):
        df = self.beamInputDirectory.eventsFile.file
        df = df.loc[df["type"] == "PersonEntersVehicle", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class ModeChoiceEvents(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "event_id"

    def load(self):
        df = self.beamInputDirectory.eventsFile.file
        df = df.loc[df["type"] == "ModeChoice", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class ModeVMT(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        pathTraversalEvents: PathTraversalEvents,
    ):
        super().__init__(outputDataDirectory)
        self.indexedOn = "pathTraversalMode"
        self.pathTraversalEvents = pathTraversalEvents

    def load(self):
        df = self.pathTraversalEvents.dataFrame.groupby("mode_extended").agg(
            {"vehicleMiles": "sum"}
        )
        df.index.name = self.indexedOn
        return df
