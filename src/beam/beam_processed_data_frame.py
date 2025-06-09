import hashlib
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.beam.beam_input_directory import BeamRunOutputDirectory
from src.constants import TMP_DIR
from src.geometry import Geometry
from src.processed_data_frame import (
    ProcessedDataFrame,
)
from src.processed_data_frame_mixins import EitherLinkStatsFile, TAZBasedDataFrame
from src.beam.beam_transformations import (
    getLinkStatsFromPathTraversals,
    fixPathTraversals,
    labelNetworkWithTaz,
    mergeLinkstatsWithNetwork,
)


class PathTraversalEvents(ProcessedDataFrame):
    """
    Represents path traversal events data extracted from an events file and then preprocessed

    Attributes:
        beamInputDirectory (BeamRunOutputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunOutputDirectory,
        *args,
        **kwargs,
    ):
        """
        Initializes a PathTraversalEvents instance from raw events file

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunOutputDirectory): The input directory for the Beam run.
        """
        super().__init__(outputDataDirectory, beamInputDirectory, *args, **kwargs)
        self.beamInputDirectory = beamInputDirectory
        # Index is set in load after read_csv
        self.indexedOn = None  # Or 'event_id' if it's consistent

    def preprocess(self, df):
        """
        Preprocesses the path traversal events DataFrame by applying fixPathTraversals.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with added columns like mode_extended and vehicleMiles.
        """
        print(
            f"Preprocessing PathTraversalEvents using fixPathTraversals ({df.shape[0]} rows)..."
        )
        # Call fixPathTraversals to add necessary columns
        processed_df = fixPathTraversals(df)
        return processed_df

    def load(self):
        """
        Loads path traversal events data from the Beam input directory, dropping unnecessary columns

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

        # Check if eventTypes dictionary already has the data loaded by collectEvents
        if "PathTraversal" not in self.beamInputDirectory.eventsFile.eventTypes:
            print(
                "Collecting PathTraversal events for {0} from {1}".format(
                    self.__class__.__name__,
                    self.beamInputDirectory.eventsFile.filePath,
                )
            )
            # collectEvents populates self.beamInputDirectory.eventsFile.eventTypes
            self.beamInputDirectory.eventsFile.collectEvents(["PathTraversal"])

        # Now access the collected data from the dictionary
        # Check if 'PathTraversal' key exists after collection attempt
        if "PathTraversal" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["PathTraversal"]
            # Indexing is typically added by collectEvents or read_csv, but ensure it's set if needed
            if df.index.name is None:
                df.index.name = "event_id"  # Assuming default index is event_id
            print(f"Loaded {df.shape[0]} raw PathTraversal events.")
            return df
        else:
            print("Failed to load PathTraversal events into eventsFile dictionary.")
            return None  # Return None if data is not available


class PersonTrips(ProcessedDataFrame):
    """
    Represents the collection of processed event dataframes (PathTraversal, ModeChoice, etc.)
    needed as input for the Inexus trip aggregation process.

    Note: The .dataFrame property of this class returns a DICTIONARY of DataFrames,
    not a single DataFrame, deviating from the standard OutputDataFrame pattern.
    Its primary purpose is to load/cache the input for runInexus.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunOutputDirectory,
        *args,
        **kwargs,
    ):
        super().__init__(outputDataDirectory, beamInputDirectory, *args, **kwargs)
        self.beamInputDirectory = beamInputDirectory
        # This class doesn't produce a single DataFrame output with a standard index.
        # Its 'dataFrame' property will return a dictionary.
        self.indexedOn = None
        self.__requiredTables = {
            "PathTraversal",
            "PersonEntersVehicle",  # May not be strictly required for Inexus agg, but in original list
            "ModeChoice",
            "ParkingEvent",
            # "actend", # Not events, need separate handling or remove from this list
            # "actstart", # Not events, need separate handling or remove from this list
            "PersonCost",
            "Replanning",
            "TeleportationEvent",
        }

    def load(self):
        raise NotImplementedError(
            "PersonTrips does not implement load(). Use dataFrame property to access data."
        )

    # Override cached, _write, _read to handle a dictionary of dataframes
    @property
    def cached(self) -> bool:
        """
        Checks if ALL required DataFrames for the dictionary are cached.
        The cache is stored as multiple parquet files with suffixes.
        """
        base_loc = self._diskLocation.replace(".parquet", "")
        # Check for cache file for each required table
        all_cached = True
        for tab in self.__requiredTables:
            file_loc = f"{base_loc}_{tab}.parquet"
            if not os.path.exists(file_loc):
                all_cached = False
                # print(f"Cache miss for {tab} at {file_loc}")
                break  # If any is missing, the whole set is not cached
            # Optional: check if file is empty or potentially corrupted?
            # For now, just checking existence.
        return all_cached

    def _write(self, obj: Dict[str, pd.DataFrame]):
        """
        Writes a dictionary of DataFrames to multiple parquet files.
        """
        assert isinstance(obj, dict)
        base_loc = self._diskLocation.replace(".parquet", "")
        # Ensure temporary directory exists
        os.makedirs(TMP_DIR, exist_ok=True)

        success = True
        for grp, df in obj.items():
            if not isinstance(df, pd.DataFrame):
                print(
                    f"Warning: Item '{grp}' in dictionary is not a DataFrame. Skipping write for this item."
                )
                continue

            file_loc = f"{base_loc}_{grp}.parquet"
            print(f"Saving processed event data for '{grp}' to {file_loc}")
            try:
                # Use default index=True, as the processed dataframes have meaningful indices (IDMerged, eventOrder)
                df.to_parquet(file_loc, engine="fastparquet")
            except Exception as e:
                print(f"OH NO! Error writing {grp} to {file_loc}: {e}")
                success = False  # Mark as failed if any write fails
                # Optionally try to clean up the failed file
                if os.path.exists(file_loc):
                    try:
                        os.remove(file_loc)
                    except:
                        pass

        # Optional: Write a sentinel file only if ALL writes were successful
        # sentinel_file = f"{base_loc}.sentinel"
        # if success:
        #      try:
        #           with open(sentinel_file, 'w') as f:
        #                f.write("Complete")
        #      except Exception as e:
        #           print(f"Warning: Failed to write sentinel file {sentinel_file}: {e}")
        # else:
        #      # If not successful, remove any existing sentinel
        #      if os.path.exists(sentinel_file):
        #           try: os.remove(sentinel_file)
        #           except: pass

    def _read(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Reads multiple parquet files into a dictionary of DataFrames.
        Returns None if any required file is missing or fails to read.
        """
        out = {}
        base_loc = self._diskLocation.replace(".parquet", "")
        read_success = True
        for tab in self.__requiredTables:
            file_loc = f"{base_loc}_{tab}.parquet"
            if not os.path.exists(file_loc):
                print(f"Required cache file missing during read: {file_loc}")
                read_success = False
                break  # Cannot read if a required file is missing

            try:
                out[tab] = pd.read_parquet(file_loc, engine="fastparquet")
            except Exception as e:
                print(
                    f"Error reading required cache file {file_loc} for table '{tab}': {e}"
                )
                read_success = False
                break  # Cannot read if any file is corrupted

        if read_success:
            print("Successfully read all required event data from cache.")
            return out
        else:
            # If read failed, clear the potentially incomplete cache set
            print("Read failed. Clearing partial cache.")
            self.clearCache()  # clearCache checks existence before removing
            return None

    # Override dataFrame property getter to handle dictionary loading/caching
    @property
    def dataFrame(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Property to lazily load the dictionary of DataFrames and return it.
        Tries reading from cache first, then loads raw events and preprocesses
        using doInexus if not available.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of processed DataFrames,
                                     or None if loading/preprocessing fails.
        """
        if self._dataFrame is None:
            print(f"Accessing dataFrame for {self.__class__.__name__}...")
            if self.cached:
                print(
                    f"Reading {self.__class__.__name__} (dictionary) from cache base {self._diskLocation.replace('.parquet', '')}..."
                )
                self._dataFrame = self._read()
                # If _read failed (e.g., missing/corrupted files), _dataFrame will still be None
                if self._dataFrame is not None:
                    print(f"Successfully read {self.__class__.__name__} from cache.")
            if self._dataFrame is None:
                print(
                    f"Cache miss or failed read for {self.__class__.__name__}. Loading raw events."
                )

                # Load raw events for all required types
                if not self.__requiredTables.issubset(
                    self.beamInputDirectory.eventsFile.eventTypes
                ):
                    print(
                        "Collecting required events for {0} from {1}".format(
                            self.__class__.__name__,
                            self.beamInputDirectory.eventsFile.filePath,
                        )
                    )
                    # collectEvents populates self.beamInputDirectory.eventsFile.eventTypes
                    self.beamInputDirectory.eventsFile.collectEvents(
                        list(self.__requiredTables)
                    )

                raw_events_dict = {}
                all_raw_loaded = True
                for tab in self.__requiredTables:
                    if tab in self.beamInputDirectory.eventsFile.eventTypes:
                        raw_events_dict[tab] = (
                            self.beamInputDirectory.eventsFile.eventTypes[tab]
                        )
                        print(
                            f"Loaded {raw_events_dict[tab].shape[0]} raw events for {tab}."
                        )
                    else:
                        print(
                            f"Warning: Raw event type '{tab}' not found after collection attempt."
                        )
                        # Decide how to handle missing raw event types.
                        # For now, add an empty dataframe and allow doInexus to handle it.
                        raw_events_dict[tab] = pd.DataFrame()  # Add empty DF

                # Ensure essential event types are present before processing
                essential_events = ["PathTraversal", "ModeChoice"]
                if not all(
                    tab in raw_events_dict and not raw_events_dict[tab].empty
                    for tab in essential_events
                ):
                    print(
                        f"Error: Essential raw event types ({essential_events}) not loaded or are empty. Cannot preprocess."
                    )
                    return None  # Cannot proceed without essential data

                print(f"Preprocessing raw event dictionary using doInexus...")
                # Preprocess the loaded raw events dictionary using doInexus
                # The preprocess method is already defined below and calls doInexus
                self._dataFrame = self.preprocess(raw_events_dict)

                if self._dataFrame is None:
                    print(f"Preprocessing PersonTrips returned None.")
                    return None  # Preprocessing failed

                print(
                    f"Preprocessing complete. Writing {self.__class__.__name__} (dictionary) to cache base {self._diskLocation.replace('.parquet', '')}..."
                )
                self._write(self._dataFrame)
                # If writing failed, _dataFrame is still set, but files might be incomplete/missing.
                # The read logic handles corrupted files by clearing cache.

        return self._dataFrame

    # Redefine the setter to also accept a dictionary
    @dataFrame.setter
    def dataFrame(self, obj: Dict[str, pd.DataFrame]):
        """
        Setter for the dataFrame property, accepts a dictionary of DataFrames.
        """
        self._dataFrame = obj

    # Removed the chunk method as it's handled in getAggregatedTrips in BeamOutputData now


class PersonEntersVehicleEvents(ProcessedDataFrame):
    """
    Represents person enters vehicle events data from the raw events file.

    Attributes:
        beamInputDirectory (BeamRunOutputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a PersonEntersVehicleEvents instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunOutputDirectory): The input directory for the Beam run.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, beamInputDirectory, *args, **kwargs)
        self.beamInputDirectory = beamInputDirectory
        # Index is set in load after read_csv
        self.indexedOn = None  # Or 'event_id' if consistent

    def load(self):
        """
        Loads person enters vehicle events data from the Beam input directory.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        # Check if eventTypes dictionary already has the data loaded by collectEvents
        if "PersonEntersVehicle" not in self.beamInputDirectory.eventsFile.eventTypes:
            print(
                "Collecting PersonEntersVehicle events from {0}".format(
                    self.beamInputDirectory.eventsFile.filePath
                )
            )
            self.beamInputDirectory.eventsFile.collectEvents(["PersonEntersVehicle"])

        # Now access the collected data from the dictionary
        if "PersonEntersVehicle" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["PersonEntersVehicle"]
            if df.index.name is None:
                df.index.name = "event_id"
            print(f"Loaded {df.shape[0]} raw PersonEntersVehicle events.")
            return df
        else:
            print("Failed to load PersonEntersVehicle events.")
            return None  # Return None if data is not available


class ModeChoiceEvents(ProcessedDataFrame):
    """
    Represents mode choice events data.

    Attributes:
        beamInputDirectory (BeamRunOutputDirectory): The input directory for the Beam run.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamInputDirectory: BeamRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, beamInputDirectory, *args, **kwargs)
        self.beamInputDirectory = beamInputDirectory
        # Index is set in load after read_csv
        self.indexedOn = None  # Or 'event_id' if consistent

    def load(self):
        # Check if eventTypes dictionary already has the data loaded by collectEvents
        if "ModeChoice" not in self.beamInputDirectory.eventsFile.eventTypes:
            print(
                "Collecting ModeChoice events from {0}".format(
                    self.beamInputDirectory.eventsFile.filePath
                )
            )
            self.beamInputDirectory.eventsFile.collectEvents(["ModeChoice"])

        # Now access the collected data from the dictionary
        if "ModeChoice" in self.beamInputDirectory.eventsFile.eventTypes:
            df = self.beamInputDirectory.eventsFile.eventTypes["ModeChoice"]
            if df.index.name is None:
                df.index.name = "event_id"
            print(f"Loaded {df.shape[0]} raw ModeChoice events.")
            return df
        else:
            print("Failed to load ModeChoice events.")
            return None  # Return None if data is not available


class RealizedModeCount(ProcessedDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        modeChoiceEvents: ModeChoiceEvents,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, modeChoiceEvents.inputDirectory, *args, **kwargs
        )
        self.modeChoiceEvents = modeChoiceEvents
        self.beamInputDirectory = (
            modeChoiceEvents.inputDirectory
        )  # Redundant, inputDirectory is already set by super
        self.indexedOn = "mode"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print(
            f"Preprocessing RealizedModeCount from ModeChoice events ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for RealizedModeCount is empty or None.")
            return pd.DataFrame(
                columns=["RealizedTrips"], index=pd.Index([], name="mode")
            )

        # Access the source data frame once, then clear memory
        # Note: The dataFrame property handles caching internally,
        # so calling clearMemory immediately after accessing it might clear
        # the cache just after it was potentially created/used.
        # It might be better to let the caching handle memory management.
        # self.modeChoiceEvents.clearMemory() # Consider if this is truly necessary/safe here

        try:
            # drop_duplicates needs the 'tripId' column. Check if it exists.
            if "tripId" in df.columns:
                # Filter to keep only the last mode choice per tripId
                df_filtered = df.drop_duplicates("tripId", keep="last")
                # Count occurrences of the 'mode' column
                if "mode" in df_filtered.columns:
                    mode_counts = df_filtered.value_counts("mode")
                    # Convert to DataFrame with column name 'RealizedTrips'
                    result_df = mode_counts.to_frame("RealizedTrips")
                    result_df.index.name = self.indexedOn  # Set index name
                    print(
                        f"Finished preprocessing RealizedModeCount ({result_df.shape[0]} rows)."
                    )
                    return result_df
                else:
                    print(
                        "Warning: 'mode' column not found in ModeChoiceEvents DataFrame."
                    )
                    return pd.DataFrame(
                        columns=["RealizedTrips"], index=pd.Index([], name="mode")
                    )

            else:
                print(
                    "Warning: 'tripId' column not found in ModeChoiceEvents DataFrame. Cannot count realized modes per trip."
                )
                return pd.DataFrame(
                    columns=["RealizedTrips"], index=pd.Index([], name="mode")
                )

        except Exception as e:
            print(f"Error during RealizedModeCount preprocessing: {e}")
            return pd.DataFrame(
                columns=["RealizedTrips"], index=pd.Index([], name="mode")
            )  # Return empty on error

    def load(self):
        """
        Loads the preprocessed ModeChoiceEvents DataFrame.
        The processing logic is in preprocess().
        """
        # Accessing self.modeChoiceEvents.dataFrame triggers its load/preprocess/cache logic
        return self.modeChoiceEvents.dataFrame


class ModeVMT(ProcessedDataFrame):
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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ModeVMT instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, pathTraversalEvents.inputDirectory, *args, **kwargs
        )
        self.indexedOn = "mode_extended"
        self.pathTraversalEvents = pathTraversalEvents

    def preprocess(self, df):
        """
        Aggregates mode vehicle miles traveled data from the preprocessed path traversal events data.

        Parameters:
            df (pd.DataFrame): The DataFrame to aggregate (preprocessed PTs).

        Returns:
            pd.DataFrame: The aggregated DataFrame.
        """
        print(f"Aggregating ModeVMT from PathTraversalEvents ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ModeVMT is empty or None.")
            return pd.DataFrame(
                columns=["vehicleMiles"], index=pd.Index([], name="mode_extended")
            )

        # Ensure required columns exist
        if "mode_extended" not in df.columns or "vehicleMiles" not in df.columns:
            print(
                "Error: Required columns 'mode_extended' or 'vehicleMiles' not found for ModeVMT aggregation."
            )
            return pd.DataFrame(
                columns=["vehicleMiles"], index=pd.Index([], name="mode_extended")
            )

        try:
            # Group by the correct mode_extended column and sum vehicleMiles
            df_agg = df.groupby("mode_extended").agg({"vehicleMiles": "sum"})
            df_agg.index.name = self.indexedOn
            print(f"Finished aggregating ModeVMT ({df_agg.shape[0]} rows).")
            return df_agg
        except Exception as e:
            print(f"Error during ModeVMT aggregation: {e}")
            return pd.DataFrame(
                columns=["vehicleMiles"], index=pd.Index([], name="mode_extended")
            )

    def load(self):
        """
        Loads the preprocessed path traversal events data.
        The aggregation logic is in preprocess().
        """
        # Accessing self.pathTraversalEvents.dataFrame triggers its load/preprocess/cache logic
        return self.pathTraversalEvents.dataFrame


class ModeVHT(ProcessedDataFrame):
    """
    Represents mode vehicle hours traveled data, calculated from the PathTraversalEvents

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ModeVMT instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, pathTraversalEvents.inputDirectory, *args, **kwargs
        )
        self.indexedOn = "mode_extended"
        self.pathTraversalEvents = pathTraversalEvents

    def preprocess(self, df):
        """
        Aggregates mode vehicle hours traveled data from the preprocessed path traversal events data.

        Parameters:
            df (pd.DataFrame): The DataFrame to aggregate (preprocessed PTs).

        Returns:
            pd.DataFrame: The aggregated DataFrame.
        """
        print(f"Aggregating ModeVHT from PathTraversalEvents ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ModeVHT is empty or None.")
            return pd.DataFrame(
                columns=["vehicleHours"], index=pd.Index([], name="mode_extended")
            )

        # Assuming 'duration' calculated in fixPathTraversals is travel time in seconds
        if "duration" not in df.columns:
            print(
                "Warning: 'duration' column not found in preprocessed PathTraversalEvents. Cannot calculate VHT."
            )
            return pd.DataFrame(
                columns=["vehicleHours"], index=pd.Index([], name="mode_extended")
            )
        if "mode_extended" not in df.columns:
            print(
                "Error: Required column 'mode_extended' not found for ModeVHT aggregation."
            )
            return pd.DataFrame(
                columns=["vehicleHours"], index=pd.Index([], name="mode_extended")
            )

        try:
            # Calculate Vehicle Hours (sum of duration in hours) per mode
            # Ensure duration column is numeric (fixPathTraversals should handle this, but double-check)
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

            # Drop rows where duration became NaN after coercion
            df_valid_duration = df.dropna(subset=["duration"])

            df_agg = df_valid_duration.groupby("mode_extended").agg(
                vehicleHours=(
                    "duration",
                    lambda x: x.sum() / 3600.0,
                )  # Assuming duration is in seconds
            )
            df_agg.index.name = self.indexedOn
            print(f"Finished aggregating ModeVHT ({df_agg.shape[0]} rows).")
            return df_agg
        except Exception as e:
            print(f"Error during ModeVHT aggregation: {e}")
            return pd.DataFrame(
                columns=["vehicleHours"], index=pd.Index([], name="mode_extended")
            )

    def load(self):
        """
        Loads the preprocessed path traversal events data.
        The aggregation logic is in preprocess().
        """
        # Accessing self.pathTraversalEvents.dataFrame triggers its load/preprocess/cache logic
        return self.pathTraversalEvents.dataFrame


class PassengerMilesByVehicleAndMode(ProcessedDataFrame):
    """
    Represents passenger miles traveled by vehicle type and mode_extended,
    calculated from the PathTraversalEvents.

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The columns to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a PassengerMilesByVehicleAndMode instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, pathTraversalEvents.inputDirectory, *args, **kwargs
        )
        self.indexedOn = [
            "vehicleType",
            "mode_extended",
        ]
        self.pathTraversalEvents = pathTraversalEvents

    def preprocess(self, df):
        """
        Loads and preprocesses passenger miles traveled data by vehicle type and mode.
        The load method already performs the necessary processing (groupby, sum, unstack).
        So, preprocess can just return the DataFrame as is.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess (aggregated data from load).

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # No additional preprocessing needed after the aggregation in load.
        return df

    def load(self):
        """
        Aggregates passenger miles traveled data by vehicle type and mode.

        Returns:
            pd.DataFrame: The aggregated and unstacked DataFrame.
        """
        print("Loading and aggregating PassengerMilesByVehicleAndMode...")
        # Access the preprocessed dataFrame directly from the source
        PTs = self.pathTraversalEvents.dataFrame.copy()
        self.pathTraversalEvents.clearMemory()

        # Ensure df is not None or empty before processing
        if PTs is None or PTs.empty:
            print(
                "Input DataFrame for PassengerMilesByVehicleAndMode is empty or None."
            )
            # Return an empty DataFrame with the expected structure
            return pd.DataFrame(
                columns=[],  # Columns will be modes after unstack
                index=pd.Index([], name="vehicleType"),
            )

        # Ensure required columns exist
        required_cols = [
            "vehicleType",
            "mode_extended",
            "currentTripMode",
            "passengerMiles",
        ]
        if not all(col in PTs.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found for PassengerMilesByVehicleAndMode aggregation."
            )
            return pd.DataFrame(columns=[], index=pd.Index([], name="vehicleType"))

        try:
            PTs.loc[
                PTs["mode_extended"].isin(["bus", "tram", "ferry", "rail", "subway"]),
                "currentTripMode",
            ] = "walk_transit"
            # Group by vehicleType and mode_extended and sum passengerMiles
            modeDistances = PTs.groupby(["vehicleType", "currentTripMode"]).agg(
                {"passengerMiles": "sum"}
            )
            # Unstack mode_extended to get modes as columns
            # Handle case where modeDistances might be empty
            if not modeDistances.empty:
                # The result of agg is a DataFrame with index ['vehicleType', 'mode_extended'] and column 'passengerMiles'.
                # Unstacking 'mode_extended' moves it from the index to columns.
                modeDistances = modeDistances["passengerMiles"].unstack(fill_value=0.0)
                # Set the index name explicitly
                modeDistances.index.name = (
                    "vehicleType"  # The remaining index after unstacking
                )
            else:
                print(
                    "Warning: Groupby result for PassengerMilesByVehicleAndMode is empty."
                )
                # Return empty DataFrame with expected index name
                modeDistances = pd.DataFrame(index=pd.Index([], name="vehicleType"))

            print(
                f"Finished aggregating PassengerMilesByVehicleAndMode ({modeDistances.shape[0]} rows)."
            )
            return modeDistances
        except Exception as e:
            print(f"Error during PassengerMilesByVehicleAndMode aggregation: {e}")
            return pd.DataFrame(index=pd.Index([], name="vehicleType"))


class ReplanningEventReasons(ProcessedDataFrame):
    """
    Represents all replanning events in BEAM

    Attributes:
        (inherits attributes from OutputDataFrame)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamRunInputDirectory: BeamRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, beamRunInputDirectory, *args, **kwargs)
        self.indexedOn = (
            None  # Index is not set in this class, often just a list of reasons
        )

    def load(self):
        """
        Loads replanning event reason data.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading ReplanningEventReasons from {self.inputDirectory.replanningEventReasonFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.inputDirectory.replanningEventReasonFile.file()
        if df is not None:
            print(f"Loaded {df.shape[0]} raw ReplanningEventReasons.")
        else:
            print("Failed to load ReplanningEventReasons.")
        return df

    def preprocess(self, df):
        """
        No specific preprocessing needed for the raw reasons file by default.
        Aggregation by reason count happens in subclasses like ReplanningEventReasonByIteration.
        """
        print(
            f"ReplanningEventReasons preprocess step (no-op). Input shape: {df.shape if df is not None else 'None'}"
        )
        return df


class ScoreStats(ProcessedDataFrame):
    """
    Keeps track of agent scores in a BEAM run

    Attributes:
        (inherits attributes from OutputDataFrame)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamRunInputDirectory: BeamRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, beamRunInputDirectory, *args, **kwargs)
        self.indexedOn = None  # Index is not set by default for this file type

    def load(self):
        """
        Loads score stats data.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading ScoreStats from {self.inputDirectory.scoreStatsFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.inputDirectory.scoreStatsFile.file()
        if df is not None:
            print(f"Loaded {df.shape[0]} raw ScoreStats.")
        else:
            print("Failed to load ScoreStats.")
        return df

    def preprocess(self, df):
        """
        No specific preprocessing needed for the raw score stats file by default.
        """
        print(
            f"ScoreStats preprocess step (no-op). Input shape: {df.shape if df is not None else 'None'}"
        )
        return df


class ModeEnergy(ProcessedDataFrame):
    """
    Represents mode energy consumption data, calculated from the PathTraversalEvents

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ModeEnergy instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, pathTraversalEvents.inputDirectory, *args, **kwargs
        )
        self.indexedOn = "mode_extended"
        self.pathTraversalEvents = pathTraversalEvents

    def preprocess(self, df):
        """
        Aggregates mode energy consumption data from the preprocessed path traversal events data.

        Parameters:
            df (pd.DataFrame): The DataFrame to aggregate (preprocessed PTs).

        Returns:
            pd.DataFrame: The aggregated DataFrame.
        """
        print(
            f"Aggregating ModeEnergy from PathTraversalEvents ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ModeEnergy is empty or None.")
            return pd.DataFrame(
                columns=["totalEnergyInJoules"],
                index=pd.Index([], name="mode_extended"),
            )

        # Ensure required columns exist
        if "mode_extended" not in df.columns or "totalEnergyInJoules" not in df.columns:
            print(
                "Error: Required columns 'mode_extended' or 'totalEnergyInJoules' not found for ModeEnergy aggregation."
            )
            return pd.DataFrame(
                columns=["totalEnergyInJoules"],
                index=pd.Index([], name="mode_extended"),
            )

        try:
            # Group by the correct mode_extended column and sum totalEnergyInJoules
            # Ensure totalEnergyInJoules column is numeric
            df["totalEnergyInJoules"] = pd.to_numeric(
                df["totalEnergyInJoules"], errors="coerce"
            ).fillna(0)

            df_agg = df.groupby("mode_extended").agg({"totalEnergyInJoules": "sum"})
            df_agg.index.name = self.indexedOn
            print(f"Finished aggregating ModeEnergy ({df_agg.shape[0]} rows).")
            return df_agg
        except Exception as e:
            print(f"Error during ModeEnergy aggregation: {e}")
            return pd.DataFrame(
                columns=["totalEnergyInJoules"],
                index=pd.Index([], name="mode_extended"),
            )

    def load(self):
        """
        Loads the preprocessed path traversal events data.
        The aggregation logic is in preprocess().
        """
        # Accessing self.pathTraversalEvents.dataFrame triggers its load/preprocess/cache logic
        return self.pathTraversalEvents.dataFrame


class ModePMT(ProcessedDataFrame):
    """
    Represents mode person miles traveled data, calculated from the PathTraversalEvents

    Attributes:
        pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        indexedOn: The column to use as the index when loading data.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,
        # nonTransitSample=0.1, # Parameter doesn't seem used, remove? Yes, removed.
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ModePMT instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(
            outputDataDirectory, pathTraversalEvents.inputDirectory, *args, **kwargs
        )
        self.indexedOn = "mode_extended"
        self.pathTraversalEvents = pathTraversalEvents

    def preprocess(self, df):
        """
        Aggregates mode person miles traveled data from the preprocessed path traversal events data.

        Parameters:
            df (pd.DataFrame): The DataFrame to aggregate (preprocessed PTs).

        Returns:
            pd.DataFrame: The aggregated DataFrame.
        """
        print(f"Aggregating ModePMT from PathTraversalEvents ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ModePMT is empty or None.")
            return pd.DataFrame(
                columns=["passengerMiles"], index=pd.Index([], name="mode_extended")
            )

        # Ensure required columns exist
        if "mode_extended" not in df.columns or "passengerMiles" not in df.columns:
            print(
                "Error: Required column 'mode_extended' or 'passengerMiles' not found for ModePMT aggregation."
            )
            return pd.DataFrame(
                columns=["passengerMiles"], index=pd.Index([], name="mode_extended")
            )

        try:
            # Group by the correct mode_extended column and sum passengerMiles
            # Ensure passengerMiles column is numeric
            df["passengerMiles"] = pd.to_numeric(
                df["passengerMiles"], errors="coerce"
            ).fillna(0)

            df_agg = df.groupby("mode_extended").agg({"passengerMiles": "sum"})
            df_agg.index.name = self.indexedOn
            print(f"Finished aggregating ModePMT ({df_agg.shape[0]} rows).")
            return df_agg
        except Exception as e:
            print(f"Error during ModePMT aggregation: {e}")
            return pd.DataFrame(
                columns=["passengerMiles"], index=pd.Index([], name="mode_extended")
            )

    def load(self):
        """
        Loads the preprocessed path traversal events data.
        The aggregation logic is in preprocess().
        """
        # Accessing self.pathTraversalEvents.dataFrame triggers its load/preprocess/cache logic
        return self.pathTraversalEvents.dataFrame


class LinkStatsFromRawFile(ProcessedDataFrame, EitherLinkStatsFile):
    """
    Represents a LinkStats dataset loaded from a raw linkstats file.
    Inherits from OutputDataFrame for caching/loading and EitherLinkStatsFile for typing and shared setup.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunOutputDirectory): The input directory.
        iteration (int): The BEAM iteration number.
        source (LinkStatsFile): The underlying raw LinkStatsFile object.
        indexedOn (List[str]): The expected index names ('link', 'hour'). (Set by mixin setup)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: "BeamRunOutputDirectory",  # More specific type
        iteration: int,  # Need iteration to get the specific LinkStatsFile
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a LinkStatsFromRawFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            inputDirectory (BeamRunOutputDirectory): The input directory for the raw linkstats file.
            iteration (int): The BEAM iteration number.
        """
        # Get the specific raw LinkStatsFile object for this iteration
        # Store it, as load() will need access to its file() method.
        self.source = inputDirectory.linkStatsFile(iteration)
        self.iteration = iteration  # Store iteration

        # Calculate the hash key based on the input directory and iteration.
        # This ensures the cache location is unique for each run/iteration of the raw file source.
        hash_key = kwargs.pop("hash_key", None)
        m = hashlib.md5()
        input_path_str = str(getattr(inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())  # Use this class's name
        m.update(str(self.iteration).encode())  # Include iteration in hash
        calculated_hash = m.hexdigest()

        # Call OutputDataFrame's __init__ using super(). The MRO is (LinkStatsFromRawFile, OutputDataFrame, EitherLinkStatsFile, object)
        # super().__init__ will call OutputDataFrame.__init__ first.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame
            inputDirectory,  # Positional for OutputDataFrame
            hash_key=calculated_hash,  # Keyword-only for OutputDataFrame
            *args,  # Pass extra args
            **kwargs,  # Pass extra kwargs
        )

        # Call the mixin's setup method *after* OutputDataFrame is initialized by super().
        self._init_either_link_stats_file()
        # self.indexedOn is now set by _init_either_link_stats_file()

    def load(self):
        """
        Loads the raw linkstats file using the source RawOutputFile object.
        This loaded data is then passed to preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print("Loading raw linkstats file from source...")
        # Access the data using the source RawOutputFile's file() method
        # The source attribute was stored in __init__.
        df = self.source.file()
        if df is not None:
            print(f"Loaded raw linkstats ({df.shape[0]} rows).")
            # Ensure index names are set correctly after loading RawOutputFile based on self.indexedOn
            # RawOutputFile might already set index_col, but names might be None.
            if (
                not isinstance(df.index, pd.MultiIndex)
                or list(df.index.names) != self.indexedOn
            ):
                print(
                    f"Warning: Raw linkstats index names are unexpected {df.index.names}. Attempting to force {self.indexedOn}"
                )
                # Assuming the structure is correct (MultiIndex with correct levels) even if names are None
                if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == len(
                    self.indexedOn
                ):
                    df.index.set_names(self.indexedOn, inplace=True)
                else:
                    print(
                        "Error: Cannot set index names on loaded raw linkstats DataFrame."
                    )
                    # Decide how to handle - return None? Proceed with default index?
                    # Let's return None as a corrupted index makes the data unusable downstream.
                    return None

        else:
            print("Failed to load raw linkstats.")
        return df  # The raw data is the data for this class

    def preprocess(self, df):
        """
        No specific preprocessing needed for the raw link stats data itself by default.
        The load method already gets the data in the desired format (indexed by link, hour).
        """
        print(
            f"LinkStatsFromRawFile preprocess step (no-op). Input shape: {df.shape if df is not None else 'None'}"
        )
        return df


class LinkStatsFromPathTraversals(ProcessedDataFrame, EitherLinkStatsFile):
    """
    Alternative linkstats file, calculated from the PathTraversalEvents.
    Inherits from OutputDataFrame for caching/loading and EitherLinkStatsFile for typing and shared setup.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunOutputDirectory): The input directory (from the PathTraversalEvents source).
        pathTraversalEvents (PathTraversalEvents): Path traversal events data source.
        iteration (int): The BEAM iteration number.
        indexedOn (List[str]): The expected index names ('link', 'hour'). (Set by mixin setup)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pathTraversalEvents: PathTraversalEvents,  # Source
        iteration: int,  # Need iteration for consistent hashing
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a LinkStatsFromPathTraversals instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            pathTraversalEvents (PathTraversalEvents): Path traversal events data.
            iteration (int): The BEAM iteration number.
        """
        # Store source object and iteration before calling super()
        self.iteration = iteration
        self.pathTraversalEvents = pathTraversalEvents
        # The input directory for this class's hashing should be the input directory of the PT source
        inputDirectory = pathTraversalEvents.inputDirectory
        self.inputDirectory = inputDirectory  # Store inputDirectory for OutputDataFrame

        # Calculate the hash key based on the input directory of the PT source and iteration.
        # This ensures the cache location is unique for each run/iteration of the calculated source.
        # Consume hash_key from kwargs before passing to super, in case it was passed by a caller.
        hash_key = kwargs.pop("hash_key", None)
        m = hashlib.md5()
        input_path_str = str(getattr(inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(str(self.iteration).encode())  # Include iteration in hash
        calculated_hash = m.hexdigest()

        # Call OutputDataFrame's __init__ using super(). The MRO is (LinkStatsFromPathTraversals, OutputDataFrame, EitherLinkStatsFile, object)
        # super().__init__ will call OutputDataFrame.__init__ first.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame
            inputDirectory,  # Positional for OutputDataFrame (get input dir from source)
            hash_key=calculated_hash,  # Keyword-only for OutputDataFrame
            *args,  # Pass extra args
            **kwargs,  # Pass extra kwargs
        )

        # Call the mixin's setup method *after* OutputDataFrame is initialized by super().
        self._init_either_link_stats_file()
        # self.indexedOn is now set by _init_either_link_stats_file()

    def load(self):
        """
        Loads the preprocessed path traversal events data from the source.
        This loaded data is then passed to preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame (preprocessed PTs).
        """
        print("Loading PathTraversalEvents for LinkStats calculation...")
        # Accessing self.pathTraversalEvents.dataFrame triggers its load/preprocess/cache logic
        df = self.pathTraversalEvents.dataFrame
        if df is not None:
            print(f"Loaded PathTraversalEvents ({df.shape[0]} rows) for LinkStats.")
        else:
            print("Failed to load PathTraversalEvents for LinkStats.")
        return df  # This DF is the input to preprocess()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the path traversal events DataFrame by applying fixPathTraversals
        and merging with network data to add link length.

        Parameters:
            df (pd.DataFrame): The DataFrame of raw path traversal events.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with link stats including length.
        """
        print(
            f"Preprocessing PathTraversalEvents using fixPathTraversals ({df.shape[0]} rows)..."
        )
        # Apply initial path traversal fixes
        processed_df = fixPathTraversals(df)

        # Ensure processed_df is not None or empty before merging
        if processed_df is None or processed_df.empty:
            print("Processed PathTraversalEvents DataFrame is empty or None after fixPathTraversals.")
            # Return an empty DataFrame with the expected structure for link stats
            return pd.DataFrame(
                columns=["traveltime", "volume"],
                index=pd.MultiIndex.from_tuples(
                    [], names=self.indexedOn
                ),
            )




        print(
            f"Merging path traversals with network data to add link length ({processed_df.shape[0]} rows)..."
        )
        # Merge with network data to add the 'length' column
        # Assuming self.beamInputDirectory.networkFile.dataFrame provides the network GeoDataFrame
        if self.inputDirectory and hasattr(self.inputDirectory, 'networkFile'):
             network_df = self.inputDirectory.networkFile.file()
             # Ensure network_df is not None or empty
             if network_df is not None and not network_df.empty:
                 # processed_df_with_length = mergeLinkstatsWithNetwork(processed_df, network_df)
                 # print(
                 #     f"Calculating link stats from merged path traversals ({processed_df_with_length.shape[0]} rows)..."
                 # )
                 # Calculate link stats from the merged path traversals
                 link_stats_df = getLinkStatsFromPathTraversals(processed_df)

                 # Ensure index names match self.indexedOn
                 if list(link_stats_df.index.names) != self.indexedOn:
                     print(
                         f"Warning: Index names from getLinkStats unexpected {list(link_stats_df.index.names)}. Expected {self.indexedOn}. Forcing set."
                     )
                     link_stats_df.index.set_names(
                         self.indexedOn, inplace=True
                     )  # Force set names

                 print(
                     f"Finished preprocessing LinkStatsFromPathTraversals ({link_stats_df.shape[0]} rows)."
                 )
                 return link_stats_df
             else:
                 print("Error: Network dataFrame is not available or empty.")
                 # Return an empty DataFrame with the expected structure
                 return pd.DataFrame(
                     columns=["traveltime", "volume"],
                     index=pd.MultiIndex.from_tuples(
                         [], names=self.indexedOn
                     ),
                 )
        else:
             print("Error: Cannot access network data through beamInputDirectory.")
             # Return an empty DataFrame with the expected structure
             return pd.DataFrame(
                 columns=["traveltime", "volume"],
                 index=pd.MultiIndex.from_tuples(
                     [], names=self.indexedOn
                 ),
             )

class LabeledNetwork(ProcessedDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamOutputData: BeamRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, beamOutputData, *args, **kwargs)
        self.beamOutputData = (
            beamOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = "linkId"

    def preprocess(self, df):
        """
        Labels the network nodes (link ends) with TAZ information.

        Parameters:
            df (pd.DataFrame): Raw network DataFrame.

        Returns:
            pd.DataFrame: Network DataFrame with TAZ labels.
        """
        print(
            f"Preprocessing LabeledNetwork using labelNetworkWithTaz ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for LabeledNetwork is empty or None.")
            return pd.DataFrame(
                columns=[
                    "toLocationX",
                    "toLocationY",
                    "linkLength",
                    "linkFreeSpeed",
                    "linkCapacity",
                    "numberOfLanes",
                    "linkModes",
                    "attributeOrigId",
                    "attributeOrigType",
                ],
                index=pd.Index([], name="linkId"),  # Match expected input index
            )
        # Ensure geometry is available
        if (
            self.beamOutputData.geometry is None
            or self.beamOutputData.geometry.gdf is None
        ):
            print("Error: Geometry is not available for labeling network with TAZ.")
            # Return input df or empty? Returning input might allow downstream steps to fail gracefully.
            return df  # Or raise error? Let's return df as is.

        # Ensure required columns exist in df
        required_cols = ["toLocationX", "toLocationY"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found in network DataFrame for labeling."
            )
            return df

        try:
            # Pass the geometry object and its TAZ index column name
            result_df = labelNetworkWithTaz(
                df,
                self.beamOutputData.geometry.gdf,
                self.beamOutputData.geometry.index,
                self.beamOutputData.geometry.crs,
            )
            print(f"Finished preprocessing LabeledNetwork ({result_df.shape[0]} rows).")
            return result_df
        except Exception as e:
            print(f"Error during LabeledNetwork preprocessing: {e}")
            return df  # Return input df on error

    def load(self):
        """
        Loads the raw network file.
        The labeling logic is in preprocess().
        """
        print(
            f"Loading raw network file from {self.beamOutputData.networkFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.beamOutputData.networkFile.file()
        if df is not None:
            print(f"Loaded raw network ({df.shape[0]} rows).")
        else:
            print("Failed to load raw network.")
        return df


class NetworkVolumesByLink(ProcessedDataFrame):
    """
    Aggregates link stats to get total VHT per link across all hours.
    Inherits from OutputDataFrame.
    Takes an EitherLinkStatsFile as source data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        source (EitherLinkStatsFile): The source LinkStats data.
        indexedOn (str): The expected index name ('link'). (Set specifically for this aggregation)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame
        source: EitherLinkStatsFile,  # Source must be an EitherLinkStatsFile
        labeledNetwork: LabeledNetwork,  # Explicitly capture LabeledNetwork
        *args,
        **kwargs,
    ):
        """
        Initializes a NetworkVolumesByLink instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            source (EitherLinkStatsFile): The source LinkStats data (raw or calculated).
        """
        # Store the source (EitherLinkStatsFile)
        self.source = source
        self.labeledNetwork = labeledNetwork

        # Calculate hash key based on the source object's hash and this class name.
        m = hashlib.md5()
        # Use input directory from source
        input_path_str = str(getattr(source.inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())  # Use this class's name
        m.update(
            source.hash().encode()
        )  # Hash based on the source's hash (OutputDataFrame's hash)
        calculated_hash = m.hexdigest()

        # Call OutputDataFrame's __init__ using super().
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame
            source.inputDirectory,  # Positional for OutputDataFrame (get input dir from source)
            hash_key=calculated_hash,  # Keyword-only for OutputDataFrame
            *args,
            **kwargs,
        )

        # The index after processing is just 'link'
        # This is specific to the aggregation in preprocess().
        self.indexedOn = (
            "link"  # This is NOT set by the mixin, it's specific to this aggregation
        )

    def load(self):
        """
        Loads the source LinkStats DataFrame (an EitherLinkStatsFile).
        This loaded data is then passed to preprocess().

        Returns:
            pd.DataFrame: The loaded LinkStats DataFrame (indexed by ['link', 'hour']).
        """
        print(
            f"Loading source LinkStats data for NetworkVolumesByLink from {self.source.__class__.__name__}..."
        )
        # Accessing self.source.dataFrame triggers its load/preprocess/cache logic
        df = self.source.dataFrame
        if df is not None:
            print(f"Loaded source LinkStats data ({df.shape[0]} rows).")
            # The source's dataFrame property should return a DataFrame indexed by self.source.indexedOn
            # which is ['link', 'hour'].
            if (
                not isinstance(df.index, pd.MultiIndex)
                or list(df.index.names) != self.source.indexedOn
            ):
                print(
                    f"Warning: Source LinkStats DataFrame index names unexpected {df.index.names}. Expected {self.source.indexedOn}."
                )
                # Attempt to force set names if the structure seems right
                if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == len(
                    self.source.indexedOn
                ):
                    print("Attempting to force set source index names.")
                    df.index.set_names(self.source.indexedOn, inplace=True)
                else:
                    print("Source DataFrame index structure is incorrect.")
                    # Decide how to handle - return None?
                    return None  # Return None if source data seems fundamentally broken

        else:
            print("Failed to load source LinkStats data.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates total VHT per link by summing VHT across all hours.
        This method receives the DataFrame loaded by load().

        Parameters:
            df (pd.DataFrame): LinkStats DataFrame (indexed by ['link', 'hour']) from self.load().

        Returns:
            pd.DataFrame: DataFrame with total VHT per link (indexed by 'link').
        """
        print(f"Preprocessing NetworkVolumesByLink ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for NetworkVolumesByLink is empty or None.")
            return pd.DataFrame(
                columns=["vht"], index=pd.Index([], name=self.indexedOn)
            )  # Use this class's indexedOn

        # Ensure required columns and index levels exist
        if "traveltime" not in df.columns or "volume" not in df.columns:
            print(
                "Error: Required columns 'traveltime' or 'volume' not found for VHT calculation."
            )
            return pd.DataFrame(
                columns=["vht"], index=pd.Index([], name=self.indexedOn)
            )
        # Check if index is a MultiIndex with expected names/levels.
        # Rely on the source's load/preprocess to set index names correctly based on its indexedOn.
        # Here, we just check the names match what we expect based on self.source.indexedOn.
        if (
            not isinstance(df.index, pd.MultiIndex)
            or list(df.index.names) != self.source.indexedOn
        ):
            print(
                f"Error: Input DataFrame for NetworkVolumesByLink does not have the expected MultiIndex {self.source.indexedOn}. Actual: {df.index.names}"
            )
            # Attempt to force fix index names based on source's expected names
            if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == len(
                self.source.indexedOn
            ):
                print("Attempting to force set index names.")
                df.index.set_names(self.source.indexedOn, inplace=True)
            else:
                print(
                    "Cannot proceed with aggregation due to incorrect index structure or names."
                )
                return pd.DataFrame(
                    columns=["vht"], index=pd.Index([], name=self.indexedOn)
                )

        try:
            # Ensure traveltime and volume are numeric
            df["traveltime"] = pd.to_numeric(df["traveltime"], errors="coerce").fillna(
                0
            )
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

            # Calculate VHT per link-hour
            df["VHT_hour"] = (
                df.traveltime * df.volume / 3600.0
            )  # traveltime likely in seconds
            df["VMT_hour"] = (
                df["volume"] * df["length"] / 1609.34
            )  # Assuming length is in meters
            df["VHT_hour_ff"] = df["length"] / df["freespeed"] * df["volume"] / 3600.0

            # Group by link (the first level of the index) and sum VHT across all hours
            # Ensure 'link' is the first level name for groupby
            if df.index.names[0] != "link":
                print(
                    f"Error: Expected 'link' as the first index level name for groupby, but got {df.index.names[0]}."
                )
                return pd.DataFrame(
                    columns=["vht"], index=pd.Index([], name=self.indexedOn)
                )

            df_agg = df.groupby(level="link").agg(
                vht=("VHT_hour", "sum"),
                vmt=("VMT_hour", "sum"),
                vht_ff=("VHT_hour_ff", "sum"),
            )
            df_agg["mph"] = df_agg["vmt"] / df_agg["vht"]
            df_agg["mph_ff"] = df_agg["vmt"] / df_agg["vht_ff"]
            df_agg["delay_h"] = df_agg["vht"] - df_agg["vht_ff"]

            df_agg.index.name = self.indexedOn  # Set index name for the output DF
            print(
                f"Finished preprocessing NetworkVolumesByLink ({df_agg.shape[0]} rows)."
            )
            return df_agg
        except Exception as e:
            print(f"Error during NetworkVolumesByLink preprocessing: {e}")
            return pd.DataFrame(
                columns=["vht"], index=pd.Index([], name=self.indexedOn)
            )

    # load method is inherited from OutputDataFrame and calls this class's load().


class LabeledLinkStatsFile(TAZBasedDataFrame):
    """
    Represents LinkStats data (from file or calculated) merged with Network TAZ labels.
    Inherits from TAZBasedDataFrame (for spatial processing).
    Takes an EitherLinkStatsFile as source data.
    MRO: LabeledLinkStatsFile -> TAZBasedDataFrame -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame
        source: EitherLinkStatsFile,  # Accept EitherLinkStatsFile as source
        labeledNetwork: LabeledNetwork,  # Consumed here for network attributes
        geometry: Geometry,  # Needed by TAZBasedDataFrame
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):
        """
        Initializes a LabeledLinkStatsFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            source (EitherLinkStatsFile): The source LinkStats data (raw or calculated).
            labeledNetwork (LabeledNetwork): Labeled network data for merging.
            geometry (Geometry): Geometry data for spatial context (used by TAZBasedDataFrame).
        """
        # Define specific attributes needed by this class
        self.labeledNetwork = labeledNetwork  # Store the labeled network
        self.source = source  # Store the source (EitherLinkStatsFile)

        input_dir_from_source = source.inputDirectory

        m = hashlib.md5()
        # Use input directory from source
        input_path_str = str(getattr(input_dir_from_source, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())  # Use this class's name
        m.update(
            source.hash().encode()
        )  # Hash based on the source's hash (OutputDataFrame's hash)
        calculated_hash = m.hexdigest()
        super().__init__(
            # Arguments needed by TAZBasedDataFrame.__init__ (outputDataDirectory, inputDirectory, *, geometry, geoIndex, hash_key=None, ...)
            outputDataDirectory=outputDataDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            inputDirectory=input_dir_from_source,  # Positional for OutputDataFrame/TAZBasedDataFrame
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            geoIndex=geometry.index,  # Keyword-only for TAZBasedDataFrame
            hash_key=calculated_hash,  # Keyword-only for OutputDataFrame (passed through TAZBasedDataFrame)
            # Pass original extra args/kwargs
            *args,
            **kwargs,
        )

        # The index after processing should be ['link', 'hour'], inherited from the source.
        # This class's preprocess method should output a DF with this index.
        # Set this class's indexedOn based on the source's indexedOn.
        self.indexedOn = source.indexedOn

    def preprocess(self, df):
        """
        Merges LinkStats data (loaded from source) with LabeledNetwork attributes.
        This method receives the DataFrame loaded by load().

        Parameters:
            df (pd.DataFrame): LinkStats DataFrame (indexed by ['link', 'hour']) from self.load().

        Returns:
            pd.DataFrame: LinkStats DataFrame with network attributes and TAZ labels, indexed by ['link', 'hour'].
        """
        print(
            f"Preprocessing LabeledLinkStatsFile using mergeLinkstatsWithNetwork ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for LabeledLinkStatsFile is empty or None.")
            # Determine expected columns from mergeLinkstatsWithNetwork + original columns
            # This is complex, return empty DF with a minimal structure
            return pd.DataFrame(
                columns=[
                    "VMT",
                    "VHT",
                    "linkLength",
                    "linkFreeSpeed",
                    "linkCapacity",
                    "numberOfLanes",
                    "linkModes",
                    "attributeOrigId",
                    "attributeOrigType",
                    self.geometry.index,
                ],
                index=pd.MultiIndex.from_tuples(
                    [], names=self.indexedOn
                ),  # Use self.indexedOn
            )

        # Ensure required columns/index exist in df (LinkStats data from source)
        linkstats_required = ["volume", "traveltime"]
        if not all(col in df.columns for col in linkstats_required):
            print(
                f"Error: Required LinkStats columns {linkstats_required} not found for merge."
            )
            return pd.DataFrame(
                columns=[
                    "VMT",
                    "VHT",
                    "linkLength",
                    "linkFreeSpeed",
                    "linkCapacity",
                    "numberOfLanes",
                    "linkModes",
                    "attributeOrigId",
                    "attributeOrigType",
                    self.geometry.index,
                ],
                index=pd.MultiIndex.from_tuples([], names=self.indexedOn),
            )
        if (
            not isinstance(df.index, pd.MultiIndex)
            or list(df.index.names) != self.indexedOn
        ):
            print(
                f"Error: Input LinkStats DataFrame does not have the expected MultiIndex {self.indexedOn}. Actual: {df.index.names}"
            )
            if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == len(
                self.indexedOn
            ):
                print("Attempting to force set index names.")
                df.index.set_names(self.indexedOn, inplace=True)
            else:
                print(
                    "Cannot proceed with merging due to incorrect index structure or names."
                )
                return pd.DataFrame(
                    columns=[
                        "VMT",
                        "VHT",
                        "linkLength",
                        "linkFreeSpeed",
                        "linkCapacity",
                        "numberOfLanes",
                        "linkModes",
                        "attributeOrigId",
                        "attributeOrigType",
                        self.geometry.index,
                    ],
                    index=pd.MultiIndex.from_tuples([], names=self.indexedOn),
                )

        # Ensure LabeledNetwork data is available
        # Access self.labeledNetwork.dataFrame which handles its own load/cache
        network_df = self.labeledNetwork.dataFrame
        if network_df is None or network_df.empty:
            print("Error: LabeledNetwork data is not available for merging.")
            return pd.DataFrame(
                columns=["VMT", "VHT"],  # Return empty if network missing
                index=df.index,  # Keep original index structure
            )

        # Ensure required columns/index exist in network_df (LabeledNetwork)
        network_required_index = "linkId"  # LabeledNetwork is indexed by linkId
        network_required_cols = [
            "linkLength",
            "linkFreeSpeed",
            "linkCapacity",
            "numberOfLanes",
            "linkModes",
            "attributeOrigId",
            "attributeOrigType",
            self.geometry.index,
        ]  # Columns expected from LabeledNetwork

        if network_df.index.name != network_required_index:
            print(
                f"Error: LabeledNetwork index name is not '{network_required_index}'."
            )
            return pd.DataFrame(columns=["VMT", "VHT"], index=df.index)
        if not all(col in network_df.columns for col in network_required_cols):
            print(
                f"Error: Required LabeledNetwork columns {network_required_cols} not found for merge."
            )
            return pd.DataFrame(columns=["VMT", "VHT"], index=df.index)

        try:
            # Ensure numeric types for calculation before merge
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["traveltime"] = pd.to_numeric(df["traveltime"], errors="coerce").fillna(
                0
            )
            if (
                "link" not in df.index.names
            ):  # Should be handled by index name check above, but defensive
                print("Error: 'link' index level name is missing for merging.")
                return pd.DataFrame(columns=["VMT", "VHT"], index=df.index)

            result_df = mergeLinkstatsWithNetwork(
                df,
                network_df,
                self.geometry.index,
            )
            print(
                f"Finished preprocessing LabeledLinkStatsFile ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during LabeledLinkStatsFile preprocessing: {e}")
            # Attempt to return something structured even on error
            # This is tricky as column presence depends on the merge.
            # Return an empty DF with expected columns.
            return pd.DataFrame(
                columns=[
                    "VMT",
                    "VHT",
                    "linkLength",
                    "linkFreeSpeed",
                    "linkCapacity",
                    "numberOfLanes",
                    "linkModes",
                    "attributeOrigId",
                    "attributeOrigType",
                    self.geometry.index,
                ],
                index=pd.MultiIndex.from_tuples([], names=self.indexedOn),
            )

    def load(self):
        """
        Loads the source LinkStats DataFrame (an EitherLinkStatsFile).
        This loaded data is then passed to preprocess().

        Returns:
            pd.DataFrame: The loaded LinkStats DataFrame (indexed by ['link', 'hour']).
        """
        print(
            f"Loading source LinkStats data for LabeledLinkStatsFile from {self.source.__class__.__name__}..."
        )
        # Accessing self.source.dataFrame triggers its load/preprocess/cache logic
        # self.source is the EitherLinkStatsFile object passed in __init__
        df = self.source.dataFrame
        if df is not None:
            print(f"Loaded source LinkStats data ({df.shape[0]} rows).")
            # The source's dataFrame property should return a DataFrame indexed by self.source.indexedOn
            # which is ['link', 'hour'].
            if (
                not isinstance(df.index, pd.MultiIndex)
                or list(df.index.names) != self.source.indexedOn
            ):
                print(
                    f"Warning: Source LinkStats DataFrame index names unexpected {df.index.names}. Expected {self.source.indexedOn}."
                )
                # Attempt to force set names if the structure seems right
                if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == len(
                    self.source.indexedOn
                ):
                    print("Attempting to force set source index names.")
                    df.index.set_names(self.source.indexedOn, inplace=True)
                else:
                    print("Source DataFrame index structure is incorrect.")
                    # Decide how to handle - return None?
                    return None  # Return None if source data seems fundamentally broken

        else:
            print("Failed to load source LinkStats data.")
        return df


class TAZTrafficVolumes(TAZBasedDataFrame):
    """
    Aggregates LabeledLinkStatsFile data to TAZ/hour/roadType level.
    Inherits from TAZBasedDataFrame for spatial processing capabilities.
    MRO: TAZTrafficVolumes -> TAZBasedDataFrame -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame
        labeledLinkStatsFile: LabeledLinkStatsFile,  # Consumed here as source data
        geometry: Geometry,  # Needed by TAZBasedDataFrame (passed up)
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):
        # Store the source data object
        self.labeledLinkStatsFile = labeledLinkStatsFile

        # The index after preprocessing is [TAZ, hour, attributeOrigType]
        # TAZBasedDataFrame uses geoIndex, which is part of this multi-index.
        # Pass geometry and geoIndex up to TAZBasedDataFrame.
        super().__init__(
            outputDataDirectory=outputDataDirectory,  # Pass to TAZBasedDataFrame/OutputDataFrame
            inputDirectory=labeledLinkStatsFile.inputDirectory,  # Pass to OutputDataFrame
            geometry=geometry,  # Pass to TAZBasedDataFrame (keyword-only)
            geoIndex=geometry.index,  # Pass the correct geo index name (keyword-only)
            *args,  # Pass any extra positional args
            **kwargs,  # Pass any extra keyword args
        )

        self.indexedOn = [
            self.geometry.index,
            "hour",
            "attributeOrigType",
        ]  # Set based on aggregation levels

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates LabeledLinkStatsFile data by TAZ, hour, and road type.

        Parameters:
            df (pd.DataFrame): LabeledLinkStatsFile DataFrame.

        Returns:
            pd.DataFrame: Aggregated DataFrame with VMT, VHT, and calculated MPH.
        """
        print(f"Preprocessing TAZTrafficVolumes ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for TAZTrafficVolumes is empty or None.")
            # Return empty DataFrame with expected columns/index levels
            expected_cols = ["VMT", "VHT", "mph"]
            expected_index_names = [self.geometry.index, "hour", "attributeOrigType"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=expected_index_names),
            )

        # Ensure required columns and index levels exist in df (LabeledLinkStatsFile)
        required_cols = [
            "VMT",
            "VHT",
        ]  # These should be calculated in LabeledLinkStatsFile.preprocess
        required_index_levels = [
            "link",
            "hour",
            self.geometry.index,
            "attributeOrigType",
        ]  # Expected levels after merge

        # Check if required columns exist (VMT, VHT should be created by LabeledLinkStatsFile preprocess)
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found in LabeledLinkStatsFile DataFrame."
            )
            expected_cols = ["VMT", "VHT", "mph"]
            expected_index_names = [self.geometry.index, "hour", "attributeOrigType"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=expected_index_names),
            )

        agg_cols = [self.geometry.index, "hour", "attributeOrigType"]
        if not all(col in (list(df.columns) + (df.index.names)) for col in agg_cols):
            print(
                f"Error: Required columns {agg_cols} not found in LabeledLinkStatsFile DataFrame for grouping."
            )
            expected_cols = ["VMT", "VHT", "mph"]
            expected_index_names = [self.geometry.index, "hour", "attributeOrigType"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=expected_index_names),
            )

        try:
            # Ensure VMT and VHT are numeric before calculation
            df["VMT"] = pd.to_numeric(df["VMT"], errors="coerce").fillna(0)
            df["VHT"] = pd.to_numeric(df["VHT"], errors="coerce").fillna(0)

            # Perform aggregation using the 'process' method inherited from TAZBasedDataFrame
            # This method handles the groupby and mapping.
            # It expects the input dataframe (df) to be passed to it *within* its own logic,
            # but here we are in the preprocess method which *receives* the dataframe.
            # So we will call the aggregation logic directly here.

            # Group by TAZ, hour, and road type, and sum VMT/VHT
            df_agg = df.groupby(agg_cols).agg({"VMT": "sum", "VHT": "sum"})

            # Calculate MPH, handling division by zero
            df_agg["mph"] = df_agg["VMT"] / df_agg["VHT"].replace(
                0, np.nan
            )  # Replace 0 VHT with NaN before division

            # Set index names explicitly after aggregation
            df_agg.index.set_names(self.indexedOn, inplace=True)

            print(f"Finished preprocessing TAZTrafficVolumes ({df_agg.shape[0]} rows).")
            return df_agg
        except Exception as e:
            print(f"Error during TAZTrafficVolumes preprocessing: {e}")
            expected_cols = ["VMT", "VHT", "mph"]
            expected_index_names = [self.geometry.index, "hour", "attributeOrigType"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=expected_index_names),
            )

    def load(self):
        """
        Loads the LabeledLinkStatsFile DataFrame.
        The aggregation logic is in preprocess().
        """
        print("Loading LabeledLinkStatsFile for TAZTrafficVolumes...")
        df = self.labeledLinkStatsFile.dataFrame
        if df is not None:
            print(
                f"Loaded LabeledLinkStatsFile ({df.shape[0]} rows) for TAZTrafficVolumes."
            )
        else:
            print("Failed to load LabeledLinkStatsFile for TAZTrafficVolumes.")
        return df


class NetworkVolumesByLinkByIteration(ProcessedDataFrame):
    """
    Aggregates NetworkVolumesByLink data across sub-iterations within a single BEAM run.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamRunInputDirectory: BeamRunOutputDirectory,
    ):
        # This class aggregates data across sub-iterations within a single BEAM run,
        # so it only needs the BEAM run's output directory.
        # pilatesInputDict is not relevant here.

        super().__init__(
            outputDataDirectory,
            beamRunInputDirectory,
        )
        # The index after processing will be ['link', 'iteration']
        self.indexedOn = ["link", "iteration"]

    def load(self) -> Optional[pd.DataFrame]:
        """
        Loads and aggregates NetworkVolumesByLink data across sub-iterations within a single BEAM run.
        """
        print(f"Loading {self.__class__.__name__} by aggregating across sub-iterations...")

        beam_run_dir = self._inputDirectory.directoryPath
        iters_dir = os.path.join(beam_run_dir, "ITERS")

        if not os.path.isdir(iters_dir):
            print(f"ITERS directory not found in {beam_run_dir}. Cannot load data.")
            return None

        aggregated_dfs = []
        # Find all iteration directories (it.X)
        iter_dirs = [d for d in os.listdir(iters_dir) if d.startswith("it.")]

        if not iter_dirs:
            print(f"No iteration directories found in {iters_dir}. Cannot load data.")
            return None

        # Sort iteration directories numerically
        iter_dirs.sort(key=lambda x: int(x.split(".")[1]))

        for iter_dir_name in iter_dirs:
            iter_match = re.match(r"it\.(\d+)", iter_dir_name)
            if not iter_match:
                continue  # Skip directories not matching the pattern

            iteration = int(iter_match.group(1))
            link_stats_path = os.path.join(
                iters_dir, iter_dir_name, f"{iteration}.linkStats.csv.gz" # Assuming gzipped CSV
            )

            if not os.path.exists(link_stats_path):
                print(f"Link stats file not found for iteration {iteration}: {link_stats_path}. Skipping.")
                continue

            try:
                # Load the link stats data for this iteration
                link_stats_df = pd.read_csv(link_stats_path)

                if link_stats_df.empty:
                    print(f"Link stats data is empty for iteration {iteration}. Skipping.")
                    continue

                # Process the data similar to NetworkVolumesByLink
                # Assuming the raw linkStats file needs aggregation
                # This logic is adapted from the original NetworkVolumesByLink load method
                if "link" not in link_stats_df.columns:
                     print(f"Warning: 'link' column not found in {link_stats_path}. Skipping iteration {iteration}.")
                     continue

                # Aggregate by link and sum relevant columns (e.g., volume, travelTime)
                # Need to confirm actual columns in linkStats.csv.gz
                # For now, assuming 'volume' and 'travelTime' as examples
                # You might need to adjust based on the actual file content
                aggregated_link_stats = link_stats_df.groupby("link").agg({
                    "volume": "sum", # Example aggregation
                    "travelTime": "mean" # Example aggregation
                    # Add other columns and aggregations as needed
                })

                # Add the iteration column
                aggregated_link_stats["iteration"] = iteration

                # Set the index to 'link' for this iteration's data
                aggregated_link_stats = aggregated_link_stats.set_index("link")

                aggregated_dfs.append(aggregated_link_stats)
                print(f"Loaded and processed link stats for iteration {iteration}.")

            except Exception as e:
                print(f"Error loading or processing link stats for iteration {iteration} from {link_stats_path}: {e}. Skipping.")
                continue

        if not aggregated_dfs:
            print("No data loaded from any iteration.")
            return None

        # Concatenate all aggregated dataframes
        final_df = pd.concat(aggregated_dfs)

        # Reset index to make 'link' and 'iteration' columns, then set multi-index
        final_df = final_df.reset_index().set_index(["link", "iteration"])


        print(f"Finished aggregation for {self.__class__.__name__} ({final_df.shape[0]} rows).")

        return final_df
