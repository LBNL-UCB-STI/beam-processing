import hashlib
import os
from typing import Dict, Tuple, List, Union, Optional
from pandas.api.types import is_numeric_dtype
import numpy as np

import pandas as pd

from src.input import (
    InputDirectory,
    BeamRunInputDirectory,
    ActivitySimRunInputDirectory,
    PilatesRunInputDirectory,
    Geometry,
    LinkStatsFile,
)
from src.transformations import (
    fixPathTraversals,
    getLinkStats,
    filterPersons,
    filterHouseholds,
    filterTrips,
    mergeLinkstatsWithNetwork,
    labelNetworkWithTaz,
    doInexus,
)


class OutputDataFrame:
    """
    Represents an output DataFrame with basic functionality like loading and preprocessing.
    Designed to load and preprocess raw BEAM outputs stored in an OutputDataDirectory.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (InputDirectory): The input directory associated with the output.
        _dataFrame (pd.DataFrame): Internal variable to store the loaded DataFrame.
        _diskLocation (str): The file location for caching the DataFrame.
        indexedOn (str): The column to use as the index when loading data.
    """

    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", inputDirectory: InputDirectory
    ):
        """
        Initializes an OutputDataFrame instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            inputDirectory (InputDirectory): The associated input directory.
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
        """
        Generates a hash based on the input and class name.

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        for s in (self.inputDirectory.directoryPath, self.__class__.__name__):
            m.update(s.encode())
        return m.hexdigest()

    @property
    def cached(self) -> bool:
        """
        Checks if the DataFrame is cached.

        Returns:
            bool: True if the DataFrame is cached, False otherwise.
        """
        return os.path.exists(self._diskLocation)

    def clearCache(self):
        """
        Clears the cached DataFrame.
        """
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

    def _write(self, obj):
        assert isinstance(obj, pd.DataFrame)
        try:
            obj.to_parquet(self._diskLocation, engine="fastparquet")
        except:
            print("STOP!!!")

    def _read(self):
        return pd.read_parquet(self._diskLocation, engine="fastparquet")

    @property
    def dataFrame(self) -> pd.DataFrame:
        """
        Property to lazily load the DataFrame and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._dataFrame is None:
            if not self.cached:
                # self.beamInputDirectory.eventsFile.filePath = "~/Downloads/1.events-maxtelework.csv.gz"
                df = self.load()
                self._dataFrame = self.preprocess(df)
                print(
                    "Writing {0} file to {1}".format(
                        self.__class__.__name__, self._diskLocation
                    )
                )
                self._write(self._dataFrame)
                # self._dataFrame.to_parquet(self._diskLocation, engine="fastparquet")
            else:
                print(
                    "Reading {0} file from {1}".format(
                        self.__class__.__name__, self._diskLocation
                    )
                )
                try:
                    self._dataFrame = self._read()
                except Exception as e:
                    print(e)
                    self.clearCache()
                    df = self.load()
                    self._dataFrame = self.preprocess(df)
                    print(
                        "Writing {0} file from {1}".format(
                            self.__class__.__name__, self._diskLocation
                        )
                    )
                    self._dataFrame.to_parquet(self._diskLocation, engine="fastparquet")
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
        """
        Adds a new column to the DataFrame based on a mapping.

        Parameters:
            mapping (dict): The mapping to use.
            fromCol (str): The source column in the index.
            toCol (str): The new column to add.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        self.dataFrame[toCol] = self.dataFrame.index.get_level_values(fromCol).map(
            mapping
        )
        return self.dataFrame

    def unstackColumn(self, col, index):
        """
        Unstacks a specified column based on the provided index.

        Parameters:
            col (str): The column to unstack.
            index: The index to use.

        Returns:
            pd.DataFrame: The unstacked DataFrame.
        """
        return self.dataFrame[col].unstack(index)

    # def process(
    #     self,
    #     normalize: Optional[str],
    #     aggregateBy: Optional[List[str]],
    #     mapping: Optional[Dict[str, str]],
    # ) -> pd.DataFrame:
    #     raise NotImplementedError("This class does not have process defined")
    #     # return self.dataFrame


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

        if "PathTraversal" not in self.beamInputDirectory.eventsFile.eventTypes:
            print(
                "Downloading events for {0} from {1}".format(
                    self.__class__.__name__,
                    self.beamInputDirectory.eventsFile.filePath,
                )
            )
            self.beamInputDirectory.eventsFile.collectEvents(
                ["PathTraversal", "PersonEntersVehicle", "ModeChoice"]
            )
        df = self.beamInputDirectory.eventsFile.eventTypes["PathTraversal"]
        # else:
        #     df = self.beamInputDirectory.eventsFile.file()
        #     df = df.loc[df["type"] == "PathTraversal", :].dropna(axis=1, how="all")
        df.index.name = "event_id"
        return df


class PersonTrips(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, beamInputDirectory)
        self.beamInputDirectory = beamInputDirectory
        self.indexedOn = "trip_id"
        self.__requiredTables = {
            "PathTraversal",
            "PersonEntersVehicle",
            "ModeChoice",
            "ParkingEvent",
            "actend",
            "actstart",
            "PersonCost",
            "Replanning",
            "ParkingEvent",
            "TeleportationEvent",
        }

    @property
    def cached(self) -> bool:
        """
        Checks if the DataFrame is cached.

        Returns:
            bool: True if the DataFrame is cached, False otherwise.
        """
        loc = self._diskLocation.replace(".parquet", "")
        mcExists = os.path.exists("{0}_{1}.parquet".format(loc, "ModeChoice"))
        ptExists = os.path.exists("{0}_{1}.parquet".format(loc, "PathTraversal"))
        return mcExists & ptExists

    def _write(self, obj):
        assert isinstance(obj, dict)
        loc = self._diskLocation.replace(".parquet", "")
        for grp, df in obj.items():
            fileLoc = "{0}_{1}.parquet".format(loc, grp)
            print("Saving {0} file to {1}".format(fileLoc, grp))
            try:
                df.to_parquet(fileLoc, engine="fastparquet")
            except Exception as e:
                print(e)
                print("OH NO!!!")

    def _read(self):
        out = dict()
        loc = self._diskLocation.replace(".parquet", "")
        for tab in list(self.__requiredTables):
            fileLoc = "{0}_{1}.parquet".format(loc, tab)
            out[tab] = pd.read_parquet(fileLoc, engine="fastparquet")
        return out

    def preprocess(self, df):
        """
        Preprocesses the path traversal events DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        return doInexus(df)

    def load(self):
        """
        Loads path traversal events data from the Beam input directory, dropping unnecessary columns

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if not self.__requiredTables.issubset(
            self.beamInputDirectory.eventsFile.eventTypes
        ):
            print(
                "Downloading events for {0} from {1}".format(
                    self.__class__.__name__,
                    self.beamInputDirectory.eventsFile.filePath,
                )
            )
            self.beamInputDirectory.eventsFile.collectEvents(
                list(self.__requiredTables)
            )
        df = {
            tab: self.beamInputDirectory.eventsFile.eventTypes[tab]
            for tab in list(self.__requiredTables)
        }

        return df

    def chunk(self, personIdToChunk):
        unSplitData = self.dataFrame
        outputData = dict()
        try:
            print(
                unSplitDf.index.get_level_values("IDMerged")
                .astype(int)
                .map(personIdToChunk)
                .head(2)
            )
        except:
            print("DSTSTS")
        for fileType in [
            "ModeChoice",
            "PathTraversal",
            "TeleportationEvent",
            "PersonCost",
            "Replanning",
            "ParkingEvent",
        ]:
            print("Breaking {0} file into chunks".format(fileType))
            unSplitDf = unSplitData[fileType]
            unSplitDf["divisionId"] = (
                unSplitDf.index.get_level_values("IDMerged")
                .astype(int)
                .map(personIdToChunk)
            )
            outputData[fileType] = {
                k: table for k, table in unSplitDf.groupby("divisionId")
            }
        return outputData


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


class ModeEnergy(OutputDataFrame):
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
        nonTransitSample=0.1,
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
            {"totalEnergyInJoules": "sum"}
        )
        df.index.name = self.indexedOn
        df.loc[df.index.astype(str).str.startswith("car"), :] *= 10
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


class LabeledNetwork(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamOutputData: BeamRunInputDirectory,
    ):
        super().__init__(outputDataDirectory, beamOutputData)
        self.beamOutputData = beamOutputData
        self.indexedOn = "linkId"

    def preprocess(self, df):
        return labelNetworkWithTaz(df, self.beamOutputData.geometry.gdf)

    def load(self):
        return self.beamOutputData.networkFile.file()


class ProcessedPersonsFile(OutputDataFrame):
    """
    Represents a processed persons file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed persons data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed persons file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunInputDirectory,
    ):
        """
        Initializes a ProcessedPersonsFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
            activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        """
        super().__init__(outputDataDirectory, activitySimOutputData)
        self.activitySimOutputData = activitySimOutputData
        self.indexedOn = "person_id"

    def preprocess(self, df):
        return filterPersons(df)

    def load(self):
        return self.activitySimOutputData.personsFile.file()


class ProcessedHouseholdsFile(OutputDataFrame):
    """
    Represents a processed households file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed households data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed households file from the ActivitySim output data directory.
    """

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
    """
    Represents a processed trips file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed trips data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed trips file from the ActivitySim output data directory.
    """

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
    """
    Represents a processed skims file derived from Pilates output data.

    This class provides functionality to load and preprocess processed skims data obtained from Pilates simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesOutputData (PilatesRunInputDirectory): The Pilates output data directory.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed skims file from the Pilates output data directory.
    """

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


class TAZBasedDataFrame(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: InputDirectory,
        geometry: Optional[Geometry],
    ):
        super().__init__(outputDataDirectory, inputDirectory)
        self.geometry = geometry
        self.geoIndex = "TAZ"  # TODO: Generalize this

    def process(
        self,
        normalize: Optional[Dict[str, str]] = None,
        aggregateBy: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        outputColumns = set((mapping or dict()).keys())
        temp = self.dataFrame.copy()
        additionalColumns = set()
        if ("area" in (normalize or dict()).values()) | (
            ("county" in (aggregateBy or [])) | ("areatype10" in (aggregateBy or []))
        ):
            if self.geometry is None:
                raise AttributeError("You need to define a geometry to do this")
            temp = (
                temp.reset_index()
                .merge(self.geometry.gdf, left_on=self.geoIndex, right_on="taz1454")
                .set_index(self.indexedOn)
            )
        if "area" in (normalize or dict()).values():
            mapping["gacres"] = "sum"
            additionalColumns.add("gacres")
        if ("county" in (aggregateBy or [])) | ("areatype10" in (aggregateBy or [])):
            grouper = list({"county", "areatype10"}.intersection(set(aggregateBy)))
            if self.indexedOn is not None:
                if isinstance(self.indexedOn, list):
                    for io in self.indexedOn:
                        grouper.append(io)
                else:
                    grouper.append(self.indexedOn)
            if self.geoIndex in grouper:
                grouper.remove(self.geoIndex)
        elif len(aggregateBy or []) > 0:
            grouper = aggregateBy
        else:
            grouper = None
        if grouper is not None:
            temp = (
                temp.reset_index()[
                    list(outputColumns) + grouper + list(additionalColumns)
                ]
                .groupby(grouper)
                .agg(mapping)
            )
            print("done")
        for col, fn in (normalize or dict()).items():
            outputColumns.add("gacres")
            if fn == "area":
                temp[col + "Density"] = temp[col].copy() / temp["gacres"]
                outputColumns.add(col + "Density")
                outputColumns.add(col)
            else:
                raise NotImplementedError(
                    "Don't have aggregation {0} implemented yet".format(fn)
                )
        return temp[list(outputColumns)]


class LabeledLinkStatsFile(TAZBasedDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        source: Union[LinkStatsFromPathTraversals, "LinkStatsFile"],
        labeledNetwork: LabeledNetwork,
        geometry: Geometry,
    ):
        self.inputDirectory = source.inputDirectory
        self.labeledNetwork = labeledNetwork
        assert isinstance(self.inputDirectory, BeamRunInputDirectory)
        self.source = source
        if isinstance(source, LinkStatsFromPathTraversals):
            super().__init__(outputDataDirectory, source.inputDirectory, geometry)
            # self._raw_df = source.dataFrame
        elif isinstance(source, LinkStatsFile):
            super().__init__(outputDataDirectory, source.inputDirectory, geometry)
            # self._raw_df = source.file()
        else:
            raise TypeError(
                "Labeling LinkStats requires either a LinkStatasFromPathTraversals or LinkStatsFile object"
            )
        self.geoIndex = "taz1454"
        self.indexedOn = ["link", "hour"]

    def load(self):
        if isinstance(self.source, LinkStatsFromPathTraversals):
            # self.source.clearCache()
            return self.source.dataFrame
            # self._raw_df = source.dataFrame
        elif isinstance(self.source, LinkStatsFile):
            return self.source.file()
        return None

    def preprocess(self, df):
        return mergeLinkstatsWithNetwork(df, self.labeledNetwork.dataFrame)

    def hash(self):
        """
        Generates a hash based on the input and class name. Also include whether we've generated it from linkstats or events

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        for s in (
            self.inputDirectory.directoryPath,
            self.__class__.__name__,
            self.source.__class__.__name__,
        ):
            m.update(s.encode())
        return m.hexdigest()


class TAZTrafficVolumes(TAZBasedDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        labeledLinkStatsFile: LabeledLinkStatsFile,
        geometry: Geometry,
    ):
        super().__init__(
            outputDataDirectory, labeledLinkStatsFile.inputDirectory, geometry
        )
        self.labeledLinkStatsFile = labeledLinkStatsFile
        self.geoIndex = "taz1454"
        self.indexedOn = ["taz1454", "hour", "attributeOrigType"]

    def load(self):
        return self.labeledLinkStatsFile.process(
            dict(),
            ["taz1454", "hour", "attributeOrigType"],
            {"VMT": "sum", "VHT": "sum"},
        )


class MandatoryLocationsByTaz(TAZBasedDataFrame):
    """
    Represents the count of mandatory locations by TAZ derived from processed persons file.

    This class provides functionality to load and preprocess the count of mandatory locations by TAZ obtained from processed persons data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        personsFile (ProcessedPersonsFile): The processed persons file.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the count of mandatory locations by TAZ from the processed persons file.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        personsFile: ProcessedPersonsFile,
        geometry: Optional[Geometry],
    ):
        super().__init__(outputDataDirectory, personsFile.inputDirectory, geometry)
        self.personsFile = personsFile

    def load(self):
        population = self.personsFile.dataFrame["TAZ"].value_counts().astype(int)
        workplaces = self.personsFile.dataFrame["work_zone_id"].value_counts()
        workplaces = workplaces.loc[workplaces.index > 0]
        workplaces.index.set_names("TAZ", inplace=True)
        return pd.concat({"population": population, "jobs": workplaces}, axis=1).fillna(
            0
        )


class TripModeCount(TAZBasedDataFrame):
    """
    Represents the count of trip modes derived from processed trips file.

    This class provides functionality to load and preprocess the count of trip modes obtained from processed trips data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the count of trip modes from the processed trips file.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        geometry: Optional[Geometry] = None,
        indexedOn: Optional[str] = None,
    ):
        super().__init__(outputDataDirectory, tripsFile.inputDirectory, geometry)
        self.geometry = geometry
        self.indexedOn = indexedOn or "trip_mode"
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
    """
    Represents the person miles traveled (PMT) for each trip mode derived from processed trips and skims files.

    This class provides functionality to load and preprocess the person miles traveled (PMT) for each trip mode obtained from processed trips and skims data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        skimsFile (ProcessedSkimsFile): The processed skims file.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the person miles traveled (PMT) for each trip mode from the processed trips and skims files.
    """

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


class MeanDistanceToWork(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        personsFile: ProcessedPersonsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile.inputDirectory)
        self.indexedOn = "TAZ"
        self.personsFile = personsFile
        self.skimsFile = skimsFile
        self.indices = ["TAZ"]

    def load(self):
        persons = self.personsFile.dataFrame.loc[
            self.personsFile.dataFrame.work_zone_id > 0, ["work_zone_id", "TAZ"]
        ]
        persons["distanceInMiles"] = (
            self.skimsFile.dataFrame["DistanceMiles"]
            .reindex(
                pd.MultiIndex.from_frame(
                    persons.rename(
                        columns={"TAZ": "origin", "work_zone_id": "destination"}
                    )
                ).values
            )
            .values
        )
        byTaz = persons.groupby(self.indices).agg(
            {"distanceInMiles": ["sum", "length"]}
        )["distanceInMiles"]
        byTaz["meanDistance"] = byTaz["sum"] / byTaz["length"]
        return byTaz["meanDistance"].to_frame()


class TripModeCountByOrigin(TripModeCount):
    """
    Represents the count of trip modes by origin derived from processed trips file.

    This class provides functionality to load and preprocess the count of trip modes by origin obtained from processed trips data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        load(): Loads the count of trip modes by origin from the processed trips file.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        geometry: Optional[Geometry] = None,
    ):
        super().__init__(outputDataDirectory, tripsFile, geometry)
        self.indices = ["trip_mode", "origin"]
        self.geoIndex = "origin"


class TripModeCountByPrimaryPurpose(TripModeCount):
    """
    Represents the count of trip modes by primary purpose derived from processed trips file.

    This class provides functionality to load and preprocess the count of trip modes by primary purpose obtained from processed trips data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        load(): Loads the count of trip modes by primary purpose from the processed trips file.
    """

    def __init__(
        self, outputDataDirectory: "OutputDataDirectory", tripsFile: ProcessedTripsFile
    ):
        super().__init__(outputDataDirectory, tripsFile)
        self.indices = ["trip_mode", "primary_purpose"]


class TripPMTByOrigin(TripPMT):
    """
    Represents the person miles traveled (PMT) for each trip mode by origin derived from processed trips and skims files.

    This class provides functionality to load and preprocess the person miles traveled (PMT) for each trip mode by origin obtained from processed trips and skims data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        skimsFile (ProcessedSkimsFile): The processed skims file.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        load(): Loads the person miles traveled (PMT) for each trip mode by origin from the processed trips and skims files.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile, skimsFile)
        self.indices = ["trip_mode", "origin"]
        self.geoIndex = "origin"


class TripPMTByPrimaryPurpose(TripPMT):
    """
    Represents the person miles traveled (PMT) for each trip mode by primary purpose derived from processed trips and skims files.

    This class provides functionality to load and preprocess the person miles traveled (PMT) for each trip mode by primary purpose obtained from processed trips and skims data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        skimsFile (ProcessedSkimsFile): The processed skims file.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        load(): Loads the person miles traveled (PMT) for each trip mode by primary purpose from the processed trips and skims files.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        skimsFile: ProcessedSkimsFile,
    ):
        super().__init__(outputDataDirectory, tripsFile, skimsFile)
        self.indices = ["trip_mode", "primary_purpose"]


class MandatoryLocationByTazByYear(TAZBasedDataFrame):
    """
    Represents the count of population and jobs by TAZ for each year derived from processed persons file.

    This class provides functionality to load and preprocess the count of population and jobs by TAZ for each year obtained from processed persons data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        personsFile (ProcessedPersonsFile): The processed persons file.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the count of population and jobs by TAZ for each year from the processed persons file.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
        geometry: Geometry,
    ):
        super().__init__(outputDataDirectory, pilatesRunInputDirectory, geometry)
        self.pilatesInputDict = pilatesInputDict
        self.__yearToDataFrame = dict()
        self.indexedOn = ["Year", "TAZ"]

    def load(self):
        """
        Loads the count of population and jobs by TAZ for each year from the processed persons file.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
        for (yr, it), data in self.pilatesInputDict.items():
            if yr not in self.__yearToDataFrame:
                self.__yearToDataFrame[yr] = data.mandatoryLocationsByTaz.dataFrame
        return pd.concat(self.__yearToDataFrame, names=["Year", "TAZ"])


class TripPMTByYear(OutputDataFrame):
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
                self.__yearToDataFrame[yr] = data.tripPMT.dataFrame
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "mode"])
        else:
            return pd.DataFrame


class TripPMTByCountyByYear(OutputDataFrame):
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
                self.__yearToDataFrame[yr] = data.tripPMTByOrigin.process(
                    normalize=dict(), aggregateBy=["county"], mapping={"count": "sum"}
                )
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "county", "mode"])
        else:
            return pd.DataFrame


class TripModeCountByYear(OutputDataFrame):
    """
    Represents the count of trip modes for each year derived from processed trips file.

    This class provides functionality to load and preprocess the count of trip modes for each year obtained from processed trips data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesRunInputDirectory (PilatesRunInputDirectory): The Pilates run input directory.
        pilatesInputDict (Dict[Tuple[int, int], "ActivitySimRunOutputData"]): A dictionary mapping years to corresponding ActivitySimRunOutputData instances.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the count of trip modes for each year from the processed trips file.
    """

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
        """
        Loads the count of trip modes for each year from the processed trips file.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
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
            return pd.concat(self.__yearToDataFrame, names=["year", "mode"])
        else:
            return pd.DataFrame


class TripModeCountByCountyByYear(OutputDataFrame):
    """
    Represents the count of trip modes for each year derived from processed trips file.

    This class provides functionality to load and preprocess the count of trip modes for each year obtained from processed trips data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesRunInputDirectory (PilatesRunInputDirectory): The Pilates run input directory.
        pilatesInputDict (Dict[Tuple[int, int], "ActivitySimRunOutputData"]): A dictionary mapping years to corresponding ActivitySimRunOutputData instances.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the count of trip modes for each year from the processed trips file.
    """

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
        """
        Loads the count of trip modes for each year from the processed trips file.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it <= self.__lastIterationPerYear[yr]:
                    continue
            else:
                self.__lastIterationPerYear[yr] = it
        for (yr, it), data in self.pilatesInputDict.items():
            if self.__lastIterationPerYear[yr] == it:
                self.__yearToDataFrame[yr] = data.tripModeCountByOrigin.process(
                    normalize=dict(), aggregateBy=["county"], mapping={"count": "sum"}
                )
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "county", "mode"])
        else:
            return pd.DataFrame


class ModeVMTByYear(OutputDataFrame):
    """
    Represents the vehicle miles traveled (VMT) for each mode by year derived from BEAM output data.

    This class provides functionality to load and preprocess the vehicle miles traveled (VMT) for each mode by year obtained from BEAM output data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesRunInputDirectory (PilatesRunInputDirectory): The Pilates run input directory.
        pilatesInputDict (Dict[Tuple[int, int], "BeamRunOutputData"]): A dictionary mapping years to corresponding BeamRunOutputData instances.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the vehicle miles traveled (VMT) for each mode by year from the BEAM output data.
    """

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
        """
        Loads the vehicle miles traveled (VMT) for each mode by year from the BEAM output data.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it >= self.__lastIterationPerYear[yr]:
                    self.__lastIterationPerYear[yr] = it
            else:
                self.__lastIterationPerYear[yr] = it
        for yr, it in self.__lastIterationPerYear.items():
            x = 2
            while x > -2:
                try:
                    data = self.pilatesInputDict[(yr, x)]
                    self.__yearToDataFrame[yr] = data.modeVMT.dataFrame
                    x = -100
                except Exception as e:
                    print(
                        "Can't find path traversals file for year {0} iteration {1}, trying the previous iteration".format(
                            yr, x
                        )
                    )
                    print(e)
                    x -= 1
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "mode"])
        else:
            return pd.DataFrame()


class ModeEnergyByYear(OutputDataFrame):
    """
    Represents the vehicle miles traveled (VMT) for each mode by year derived from BEAM output data.

    This class provides functionality to load and preprocess the vehicle miles traveled (VMT) for each mode by year obtained from BEAM output data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesRunInputDirectory (PilatesRunInputDirectory): The Pilates run input directory.
        pilatesInputDict (Dict[Tuple[int, int], "BeamRunOutputData"]): A dictionary mapping years to corresponding BeamRunOutputData instances.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        load(): Loads the vehicle miles traveled (VMT) for each mode by year from the BEAM output data.
    """

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
        """
        Loads the vehicle miles traveled (VMT) for each mode by year from the BEAM output data.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it >= self.__lastIterationPerYear[yr]:
                    self.__lastIterationPerYear[yr] = it
            else:
                self.__lastIterationPerYear[yr] = it
        for yr, it in self.__lastIterationPerYear.items():
            x = it
            while x > -2:
                try:
                    data = self.pilatesInputDict[(yr, x)]
                    self.__yearToDataFrame[yr] = data.modeEnergy.dataFrame
                    x = -100
                except Exception as e:
                    print(
                        "Can't find path traversals file for year {0} iteration {1}, trying the previous iteration".format(
                            yr, x
                        )
                    )
                    print(e)
                    x -= 1
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "mode"])
        else:
            return pd.DataFrame()


class CongestionInfoByYear(OutputDataFrame):
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
        """
        Loads the vehicle miles traveled (VMT) for each mode by year from the BEAM output data.

        Returns:
            pd.DataFrame: The DataFrame containing the loaded data.
        """
        for (yr, it), data in self.pilatesInputDict.items():
            if yr in self.__lastIterationPerYear:
                if it >= self.__lastIterationPerYear[yr]:
                    self.__lastIterationPerYear[yr] = it
            else:
                self.__lastIterationPerYear[yr] = it
        for yr, it in self.__lastIterationPerYear.items():
            x = it
            while x > -2:
                try:
                    data = self.pilatesInputDict[(yr, x)]
                    df = data.tazTrafficVolumes.dataFrame
                    df["mph"] = df["VMT"] / df["VHT"]
                    df["congestedHours"] = df["mph"] < 1.0
                    df = df.groupby(["taz1454", "attributeOrigType"]).agg(
                        {"VMT": "sum", "VHT": "sum", "congestedHours": "sum"}
                    )
                    df["mph"] = df["VMT"] / df["VHT"]
                    self.__yearToDataFrame[yr] = df.unstack("taz1454")
                    x = -100
                except Exception as e:
                    print(
                        "Can't find path traversals file for year {0} iteration {1}, trying the previous iteration".format(
                            yr, x
                        )
                    )
                    print(e)
                    x -= 1
        if any(self.__yearToDataFrame):
            return pd.concat(self.__yearToDataFrame, names=["year", "mode"])
        else:
            return pd.DataFrame()
