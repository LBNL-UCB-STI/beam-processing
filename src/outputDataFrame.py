import gc
import hashlib
import os
from typing import Dict, Tuple, List, Union, Optional, Callable, Any
import numpy as np
import geopandas as gpd

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
    filterTours,
    # Import assignTripIdToEvents for runInexus logic
    # Import mergeWithTripsAndAggregate for runInexus logic
)

# --- Constants ---
TMP_DIR = ".tmp"
CONGESTION_THRESHOLD_MPH = 2.0
FUEL_CONVERSION_JOULES_GALLON_GASOLINE = 8.3141841e-9  # Should be MJ/gallon * J/MJ # This was incorrect, 1 MJ = 1e6 J, 1 Gallon Gasoline ~ 120 MJ. 1 Joule = 1e-6 MJ. (1 gal * 120 MJ/gal * 1e6 J/MJ)^-1 ~ 8.33e-9 gal/Joule. The factor seems correct for JOULES TO GALLONS conversion.
FUEL_CONVERSION_JOULES_GALLON_DIESEL = (
    8.3141841e-9  # MJ/gallon * J/MJ # Same note as above.
)
FUEL_CONVERSION_JOULES_KWH = 3.6e6  # J/kWh
ELECTRICITY_EMISSION_KG_PER_KWH = (
    0.0005  # kg CO2 / kWh (Example value, specific to region/grid)
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
        _hash_key (Optional[str]): Optional custom hash key provided by subclass.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: InputDirectory,
        hash_key: Optional[str] = None,  # Made keyword-only
        *args,
        **kwargs,
    ):
        """
        Initializes an OutputDataFrame instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            inputDirectory (InputDirectory): The associated input directory.
            hash_key (Optional[str]): A pre-calculated hash key from a subclass.
        """
        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        self._dataFrame = None
        self._hash_key = hash_key  # Store provided hash key

        # Use the provided hash_key if available, otherwise use default hash()
        calculated_hash = self._hash_key if self._hash_key is not None else self.hash()

        self._diskLocation = os.path.join(
            TMP_DIR,
            calculated_hash + ".parquet",
        )
        self.indexedOn = None
        # Pass remaining args/kwargs up the MRO chain
        super().__init__(*args, **kwargs)

    def hash(self):
        """
        Generates a hash based on the input and class name.
        Subclasses like InfoByYear should override this or provide hash_key in __init__.

        Returns:
            str: The generated hash.
        """
        # If a custom hash key was provided, use it
        if self._hash_key is not None:
            return self._hash_key

        # Otherwise, calculate default hash
        m = hashlib.md5()
        # Ensure inputDirectory.directoryPath exists and is a string
        input_path_str = str(getattr(self.inputDirectory, "directoryPath", ""))
        class_name_str = self.__class__.__name__

        m.update(input_path_str.encode())
        m.update(class_name_str.encode())
        return m.hexdigest()

    @property
    def cached(self) -> bool:
        """
        Checks if the DataFrame is cached.

        Returns:
            bool: True if the DataFrame is cached, False otherwise.
        """
        # Ensure temporary directory exists before checking cache file
        os.makedirs(TMP_DIR, exist_ok=True)
        return os.path.exists(self._diskLocation)

    def clearCache(self):
        """
        Clears the cached DataFrame.
        """
        if self.cached:
            try:
                os.remove(self._diskLocation)
                print(f"Cleared cache file: {self._diskLocation}")
            except OSError as e:
                print(f"Error clearing cache file {self._diskLocation}: {e}")
        self._dataFrame = None

    def clearMemory(self):
        if self._dataFrame is not None:
            print(f"Clearing memory for {self.__class__.__name__}!")
            del self._dataFrame
            self._dataFrame = None
            gc.collect()

    def load(self):
        """
        Abstract method for loading data into a DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        raise NotImplementedError("Subclasses must implement 'load'")

    def _write(self, obj):
        assert isinstance(obj, pd.DataFrame)
        try:
            # Ensure temporary directory exists before writing
            os.makedirs(TMP_DIR, exist_ok=True)
            obj.to_parquet(self._diskLocation, engine="fastparquet")
        except Exception as e:
            print(
                f"Error writing {self.__class__.__name__} to {self._diskLocation}: {e}"
            )
            # Optionally remove incomplete file if writing fails
            if os.path.exists(self._diskLocation):
                try:
                    os.remove(self._diskLocation)
                except:
                    pass

    def _read(self):
        # Add error handling if tmp dir doesn't exist (unlikely but safe)
        if not os.path.exists(TMP_DIR):
            # This case should ideally be prevented by .cached check and _write logic,
            # but adding a check here for robustness.
            print(f"Temporary directory {TMP_DIR} does not exist during read attempt.")
            return None  # Indicate read failure

        if not os.path.exists(self._diskLocation):
            # This case might happen if cache was cleared between .cached check and _read,
            # or if .cached incorrectly returned True.
            print(
                f"Cache file does not exist at {self._diskLocation} during read attempt."
            )
            return None  # Indicate read failure

        try:
            return pd.read_parquet(self._diskLocation, engine="fastparquet")
        except Exception as e:
            print(
                f"Error reading {self.__class__.__name__} from {self._diskLocation}: {e}"
            )
            # If read fails, assume cache is corrupted and clear it
            self.clearCache()
            return None  # Indicate read failure

    @property
    def dataFrame(self) -> Optional[pd.DataFrame]:
        """
        Property to lazily load the DataFrame and return it.
        Tries reading from cache first, then loads and caches if not available.

        Returns:
            pd.DataFrame: The loaded and preprocessed DataFrame, or None if loading/preprocessing fails.
        """
        if self._dataFrame is None:
            print(f"Accessing dataFrame for {self.__class__.__name__}...")
            if self.cached:
                print(
                    f"Reading {self.__class__.__name__} from cache: {self._diskLocation}"
                )
                self._dataFrame = self._read()
                # If read failed (e.g., corrupted cache), _dataFrame will still be None
                if self._dataFrame is not None:
                    print(f"Successfully read {self.__class__.__name__} from cache.")
            if self._dataFrame is None:
                print(
                    f"Cache miss or failed read for {self.__class__.__name__}. Loading raw data."
                )
                df = self.load()
                if df is None:
                    print(
                        f"Loading raw data for {self.__class__.__name__} returned None."
                    )
                    return None  # Loading failed

                print(
                    f"Preprocessing raw data for {self.__class__.__name__} ({df.shape[0]} rows)..."
                )
                self._dataFrame = self.preprocess(df)
                if self._dataFrame is None:
                    print(f"Preprocessing {self.__class__.__name__} returned None.")
                    return None  # Preprocessing failed

                print(
                    f"Preprocessing complete. Writing {self.__class__.__name__} ({self._dataFrame.shape[0]} rows) to cache: {self._diskLocation}"
                )
                self._write(self._dataFrame)
                # If writing failed, _dataFrame is still set, but the file might be incomplete/missing.
                # The read logic handles corrupted files by clearing cache.

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
        Subclasses must implement this.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        # Default implementation just returns the dataframe.
        # Subclasses are expected to override this.
        print(
            f"No custom preprocessing defined for {self.__class__.__name__}. Returning raw data."
        )
        return df

    def toCsv(self):
        """
        Saves the DataFrame to a CSV file in the output data directory.
        """
        name = self.__class__.__name__
        output_dir = self.outputDataDirectory.path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, name + ".csv")
        print(f"Saving {name} to {output_path}")
        # Accessing .dataFrame triggers load/cache if needed
        df_to_save = self.dataFrame
        if df_to_save is not None and not df_to_save.empty:
            try:
                df_to_save.to_csv(output_path)
                print(f"Successfully saved {name} to CSV.")
            except Exception as e:
                print(f"Error saving {name} to {output_path}: {e}")
        else:
            print(
                f"No dataFrame available or dataFrame is empty for {name}. Not saving CSV."
            )

    def addMapping(self, mapping: dict, fromCol: str, toCol: str):
        """
        Adds a new column to the DataFrame based on a mapping.

        Parameters:
            mapping (dict): The mapping to use.
            fromCol (str): The source column in the index (level name).
            toCol (str): The new column to add.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        # Accessing dataFrame triggers load/cache
        df = self.dataFrame
        if df is not None:
            try:
                # Check if fromCol is in index levels
                if fromCol in df.index.names:
                    df[toCol] = df.index.get_level_values(fromCol).map(mapping)
                else:
                    print(
                        f"Warning: Column '{fromCol}' not found in DataFrame index levels."
                    )
                    # Optionally try mapping from a column if not in index
                    if fromCol in df.columns:
                        df[toCol] = df[fromCol].map(mapping)
                    else:
                        print(
                            f"Warning: Column '{fromCol}' not found in DataFrame columns either. Cannot add mapping."
                        )
                        # Add column with None or NaN?
                        df[toCol] = None  # Or np.nan or pd.NA depending on dtype
                return df
            except Exception as e:
                print(f"Error applying mapping to {self.__class__.__name__}: {e}")
                return df  # Return dataframe even on error
        return None  # Return None if dataFrame was None

    def unstackColumn(self, col, index):
        """
        Unstacks a specified column based on the provided index (level name).

        Parameters:
            col (str): The column to unstack.
            index: The level name in the index to unstack by.

        Returns:
            pd.DataFrame: The unstacked DataFrame, or None if dataFrame is None or unstacking fails.
        """
        df = self.dataFrame
        if df is not None:
            try:
                # Select the column first, then unstack
                if col in df.columns:
                    return df[col].unstack(index)
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame columns.")
                    return None
            except Exception as e:
                print(
                    f"Error unstacking column '{col}' by index '{index}' for {self.__class__.__name__}: {e}"
                )
                return None  # Indicate unstacking failure
        return None  # Return None if dataFrame was None

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
        *args,
        **kwargs,
    ):
        """
        Initializes a PathTraversalEvents instance from raw events file

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        """
        super().__init__(outputDataDirectory, beamInputDirectory, *args, **kwargs)
        self.beamInputDirectory = beamInputDirectory
        # Index is set in load after read_csv
        self.indexedOn = None  # Or 'event_id' if it's consistent

    def preprocess(self, df):
        """
        Preprocesses the path traversal events DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"Preprocessing PathTraversalEvents using fixPathTraversals ({df.shape[0]} rows)..."
        )
        return fixPathTraversals(df)

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


class PersonTrips(OutputDataFrame):
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
        beamInputDirectory: BeamRunInputDirectory,
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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a PersonEntersVehicleEvents instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
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


class RealizedModeCount(OutputDataFrame):
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


class ModeVHT(OutputDataFrame):
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


class PassengerMilesByVehicleAndMode(OutputDataFrame):
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


class ReplanningEventReasons(OutputDataFrame):
    """
    Represents all replanning events in BEAM

    Attributes:
        (inherits attributes from OutputDataFrame)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamRunInputDirectory: BeamRunInputDirectory,
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


class ScoreStats(OutputDataFrame):
    """
    Keeps track of agent scores in a BEAM run

    Attributes:
        (inherits attributes from OutputDataFrame)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamRunInputDirectory: BeamRunInputDirectory,
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


class ModeEnergy(OutputDataFrame):
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


class ModePMT(OutputDataFrame):
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


class EitherLinkStatsFile(object):
    """
    Mixin class that marks a DataFrame as a source of LinkStats data,
    either from a raw file or calculated from PathTraversalEvents.
    Classes inheriting from this should also inherit from OutputDataFrame.
    Provides common LinkStats-related attributes and methods.
    """

    def _init_either_link_stats_file(self):
        """Internal setup method for the EitherLinkStatsFile mixin."""
        # This attribute is expected on classes that are EitherLinkStatsFile
        self.indexedOn = ["link", "hour"]

    def write_specific_format(self, path: str):
        """
        Placeholder method for writing LinkStats data in a specific format.
        Subclasses can override this. Default implementation saves as CSV using OutputDataFrame's method.
        """
        print(
            f"Attempting to write {self.__class__.__name__} to {path} in specific format..."
        )
        # Accessing dataFrame triggers load/cache if needed
        df_to_save = (
            self.dataFrame
        )  # Use the .dataFrame property which handles load/preprocess/cache

        if df_to_save is not None and not df_to_save.empty:
            try:
                # Use the base class's toCsv method or a specific implementation
                # For a truly *specific* format, subclasses should override this.
                # This default uses OutputDataFrame.toCsv's internal logic.
                df_to_save.to_csv(path)
                print(
                    f"Successfully saved {self.__class__.__name__} to {path} (default CSV format)."
                )
            except Exception as e:
                print(f"Error saving {self.__class__.__name__} to {path}: {e}")
        else:
            print(
                f"No dataFrame available or dataFrame is empty for {self.__class__.__name__}. Not saving."
            )
        # Subclasses should override this for non-CSV formats or custom CSV structures


class LinkStatsFromRawFile(OutputDataFrame, EitherLinkStatsFile):
    """
    Represents a LinkStats dataset loaded from a raw linkstats file.
    Inherits from OutputDataFrame for caching/loading and EitherLinkStatsFile for typing and shared setup.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunInputDirectory): The input directory.
        iteration (int): The BEAM iteration number.
        source (LinkStatsFile): The underlying raw LinkStatsFile object.
        indexedOn (List[str]): The expected index names ('link', 'hour'). (Set by mixin setup)
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: "BeamRunInputDirectory",  # More specific type
        iteration: int,  # Need iteration to get the specific LinkStatsFile
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a LinkStatsFromRawFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            inputDirectory (BeamRunInputDirectory): The input directory for the raw linkstats file.
            iteration (int): The BEAM iteration number.
        """
        # Get the specific raw LinkStatsFile object for this iteration
        # Store it, as load() will need access to its file() method.
        self.source = inputDirectory.linkStatsFile(iteration)
        self.iteration = iteration  # Store iteration

        # Calculate the hash key based on the input directory and iteration.
        # This ensures the cache location is unique for each run/iteration of the raw file source.
        hash_key = kwargs.pop('hash_key', None)
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


class LinkStatsFromPathTraversals(OutputDataFrame, EitherLinkStatsFile):
    """
    Alternative linkstats file, calculated from the PathTraversalEvents.
    Inherits from OutputDataFrame for caching/loading and EitherLinkStatsFile for typing and shared setup.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunInputDirectory): The input directory (from the PathTraversalEvents source).
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
        hash_key = kwargs.pop('hash_key', None)
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
        Calculates link stats (volumes, travel times) from preprocessed path traversal events.
        This method receives the DataFrame loaded by load().

        Parameters:
            df (pd.DataFrame): Preprocessed PathTraversalEvents DataFrame.

        Returns:
            pd.DataFrame: DataFrame of link volumes and travel times per hour (indexed by ['link', 'hour']).
        """
        print(
            f"Preprocessing LinkStatsFromPathTraversals using getLinkStats ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for LinkStatsFromPathTraversals is empty or None.")
            # Return an empty DataFrame with the expected structure
            return pd.DataFrame(
                columns=["traveltime", "volume"],
                index=pd.MultiIndex.from_tuples(
                    [], names=self.indexedOn
                ),  # Use self.indexedOn set by mixin setup
            )
        try:
            result_df = getLinkStats(df)
            # The getLinkStats function already sets index names ['link', 'hour'].
            # Ensure it matches self.indexedOn, although they should be the same.
            if list(result_df.index.names) != self.indexedOn:
                print(
                    f"Warning: Index names from getLinkStats unexpected {list(result_df.index.names)}. Expected {self.indexedOn}. Forcing set."
                )
                result_df.index.set_names(
                    self.indexedOn, inplace=True
                )  # Force set names

            print(
                f"Finished preprocessing LinkStatsFromPathTraversals ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during LinkStatsFromPathTraversals preprocessing: {e}")
            return pd.DataFrame(
                columns=["traveltime", "volume"],
                index=pd.MultiIndex.from_tuples(
                    [], names=self.indexedOn
                ),  # Use self.indexedOn set by mixin setup
            )


class LabeledNetwork(OutputDataFrame):
    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        beamOutputData: BeamRunInputDirectory,
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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ProcessedPersonsFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
            activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        """
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, activitySimOutputData, *args, **kwargs)
        self.activitySimOutputData = (
            activitySimOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = "person_id"

    def preprocess(self, df):
        """
        Preprocesses the raw persons DataFrame using filterPersons.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess (raw persons file).

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"Preprocessing ProcessedPersonsFile using filterPersons ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ProcessedPersonsFile is empty or None.")
            # Return empty DataFrame with expected columns (defined in filterPersons)
            expected_cols = [
                "earning",
                "worker",
                "student",
                "household_id",
                "school_zone_id",
                "age",
                "work_zone_id",
                "TAZ",
                "home_x",
                "home_y",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="person_id")
            )

        try:
            result_df = filterPersons(df)
            print(
                f"Finished preprocessing ProcessedPersonsFile ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during ProcessedPersonsFile preprocessing: {e}")
            expected_cols = [
                "earning",
                "worker",
                "student",
                "household_id",
                "school_zone_id",
                "age",
                "work_zone_id",
                "TAZ",
                "home_x",
                "home_y",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="person_id")
            )

    def load(self):
        """
        Loads the raw persons file from the ActivitySim input directory.
        The filtering logic is in preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading raw persons file from {self.activitySimOutputData.personsFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.activitySimOutputData.personsFile.file()
        if df is not None:
            print(f"Loaded raw persons file ({df.shape[0]} rows).")
        else:
            print("Failed to load raw persons file.")
        return df


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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, activitySimOutputData, *args, **kwargs)
        self.activitySimOutputData = (
            activitySimOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = "household_id"

    def preprocess(self, df):
        """
        Preprocesses the raw households DataFrame using filterHouseholds.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess (raw households file).

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"Preprocessing ProcessedHouseholdsFile using filterHouseholds ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ProcessedHouseholdsFile is empty or None.")
            # Return empty DataFrame with expected columns (defined in filterHouseholds)
            expected_cols = [
                "recent_mover",
                "num_workers",
                "sf_detached",
                "tenure",
                "race_of_head",
                "income",
                "block_id",
                "cars",
                "hhsize",
                "TAZ",
                "num_drivers",
                "num_children",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="household_id")
            )

        try:
            result_df = filterHouseholds(df)
            print(
                f"Finished preprocessing ProcessedHouseholdsFile ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during ProcessedHouseholdsFile preprocessing: {e}")
            expected_cols = [
                "recent_mover",
                "num_workers",
                "sf_detached",
                "tenure",
                "race_of_head",
                "income",
                "block_id",
                "cars",
                "hhsize",
                "TAZ",
                "num_drivers",
                "num_children",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="household_id")
            )

    def load(self):
        """
        Loads the raw households file from the ActivitySim input directory.
        The filtering logic is in preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading raw households file from {self.activitySimOutputData.householdsFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.activitySimOutputData.householdsFile.file()
        if df is not None:
            print(f"Loaded raw households file ({df.shape[0]} rows).")
        else:
            print("Failed to load raw households file.")
        return df


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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, activitySimOutputData, *args, **kwargs)
        self.activitySimOutputData = (
            activitySimOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = "trip_id"

    def preprocess(self, df):
        """
        Preprocesses the raw trips DataFrame using filterTrips.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess (raw trips file).

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"Preprocessing ProcessedTripsFile using filterTrips ({df.shape[0]} rows)..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for ProcessedTripsFile is empty or None.")
            # Return empty DataFrame with expected columns (defined in filterTrips)
            expected_cols = [
                "person_id",
                "household_id",
                "tour_id",
                "primary_purpose",
                "purpose",
                "destination",
                "origin",
                "destination_logsum",
                "depart",
                "trip_mode",
                "mode_choice_logsum",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="trip_id")
            )

        try:
            result_df = filterTrips(df)
            print(
                f"Finished preprocessing ProcessedTripsFile ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during ProcessedTripsFile preprocessing: {e}")
            expected_cols = [
                "person_id",
                "household_id",
                "tour_id",
                "primary_purpose",
                "purpose",
                "destination",
                "origin",
                "destination_logsum",
                "depart",
                "trip_mode",
                "mode_choice_logsum",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="trip_id")
            )

    def load(self):
        """
        Loads the raw trips file from the ActivitySim input directory.
        The filtering logic is in preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading raw trips file from {self.activitySimOutputData.tripsFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.activitySimOutputData.tripsFile.file()
        if df is not None:
            print(f"Loaded raw trips file ({df.shape[0]} rows).")
        else:
            print("Failed to load raw trips file.")
        return df


class ProcessedToursFile(OutputDataFrame):
    """
    Represents a processed tours file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed tours data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunInputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed tours file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunInputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, activitySimOutputData, *args, **kwargs)
        self.activitySimOutputData = (
            activitySimOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = "tour_id"

    def preprocess(self, df):
        """
        Preprocesses the raw tours DataFrame using filterTours.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess (raw tours file).

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"Preprocessing ProcessedToursFile using filterTours ({df.shape[0]} rows if not None)..."
        )
        # filterTours handles None input, but check for empty after load
        if df is not None and df.empty:
            print("Input DataFrame for ProcessedToursFile is empty.")
            # Return empty DataFrame with expected columns (defined in filterTours)
            expected_cols = [
                "person_id",
                "tour_type",
                "tour_category",
                "number_of_participants",
                "destination",
                "origin",
                "household_id",
                "start",
                "end",
                "duration",
                "composition",
                "destination_logsum",
                "tour_mode",
                "mode_choice_logsum",
                "atwork_subtour_frequency",
                "parent_tour_id",
                "stop_frequency",
                "primary_purpose",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="tour_id")
            )
        try:
            result_df = filterTours(df)  # filterTours returns empty DF for None input
            if result_df is not None:
                print(
                    f"Finished preprocessing ProcessedToursFile ({result_df.shape[0]} rows)."
                )
            else:
                print("filterTours returned None.")
            return result_df
        except Exception as e:
            print(f"Error during ProcessedToursFile preprocessing: {e}")
            expected_cols = [
                "person_id",
                "tour_type",
                "tour_category",
                "number_of_participants",
                "destination",
                "origin",
                "household_id",
                "start",
                "end",
                "duration",
                "composition",
                "destination_logsum",
                "tour_mode",
                "mode_choice_logsum",
                "atwork_subtour_frequency",
                "parent_tour_id",
                "stop_frequency",
                "primary_purpose",
            ]
            return pd.DataFrame(
                columns=expected_cols, index=pd.Index([], name="tour_id")
            )

    def load(self):
        """
        Loads the raw tours file from the ActivitySim input directory.
        The filtering logic is in preprocess().

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        print(
            f"Loading raw tours file from {self.activitySimOutputData.toursFile.filePath}..."
        )
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.activitySimOutputData.toursFile.file()
        if df is not None:
            print(f"Loaded raw tours file ({df.shape[0]} rows).")
        else:
            print("Failed to load raw tours file.")
        return df


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
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        # Pass args/kwargs up to OutputDataFrame
        super().__init__(outputDataDirectory, pilatesOutputData, *args, **kwargs)
        self.pilatesOutputData = (
            pilatesOutputData  # Redundant, inputDirectory is already set by super
        )
        self.indexedOn = ["Origin", "Destination"]

    def preprocess(self, df):
        """
        No specific preprocessing needed for the skims file by default.
        """
        print(
            f"ProcessedSkimsFile preprocess step (no-op). Input shape: {df.shape if df is not None else 'None'}"
        )
        return df

    def load(self):
        """
        Loads the skims file (OMX) from the Pilates input directory.
        The processing logic (extracting matrices) is in RawOutputFile.file().
        """
        print(f"Loading skims file from {self.pilatesOutputData.skims.filePath}...")
        # Accessing .file() on RawOutputFile triggers its load/cache logic
        df = self.pilatesOutputData.skims.file()
        if df is not None:
            print(f"Loaded skims file ({df.shape[0]} rows).")
        else:
            print("Failed to load skims file.")
        return df


class TAZBasedDataFrame(OutputDataFrame):
    """
    Mixin class for DataFrames that have a TAZ-like geographic index.
    Provides methods for merging with geometry and performing spatial aggregations.

    Attributes:
        geometry (Geometry): The geometry object for spatial operations.
        geoIndex (str): The name of the column in the DataFrame index and geometry gdf
                        that represents the geographic zone (e.g., 'TAZ', 'zoneid').
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Expected by OutputDataFrame
        inputDirectory: InputDirectory,  # Expected by OutputDataFrame
        geometry: Geometry,
        geoIndex: Optional[str] = "TAZ",
        *args,
        **kwargs,  # Accepts args for OutputDataFrame and other parents in MI
    ):
        # Consume TAZBasedDataFrame's specific keyword-only arguments
        self.geometry = geometry
        self.geoIndex = geoIndex

        # Pass outputDataDirectory and inputDirectory (positional) to OutputDataFrame,
        # along with any other *args and **kwargs received.
        super().__init__(outputDataDirectory, inputDirectory, *args, **kwargs)

        # Note: self.inputDirectory and self.outputDataDirectory are set by OutputDataFrame.__init__

    # The preprocess method in TAZBasedDataFrame is the one used for its own processing tasks,
    # like merging with geometry *if* the subclass load() returns data that needs
    # geometry merging *at this stage*. However, often the geometry merging/processing
    # is done within the load or preprocess of the specific subclass (like LabeledNetwork)
    # or within an accessor for aggregated classes like InfoByYear/Iteration.
    # Let's keep the default preprocess which does nothing unless overridden.
    # The `process` method below is where the geometry-aware aggregation happens.
    # def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
    #      # Default implementation passes through
    #      return df

    def countsInColumn(
        self, df: pd.DataFrame, column: str, nonNegative=True
    ) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        out = pd.to_numeric(df[column], errors="coerce").value_counts()
        if nonNegative:
            out = out.loc[out.index > 0]
        out.index.set_names(self.geoIndex, inplace=True)
        return out

    def setGeometry(self, geom: Geometry):
        """Sets the geometry object for this DataFrame."""
        self.geometry = geom
        # Note: Changing geometry after initialization doesn't update cache location.
        # Might need to clear cache or re-initialize if geometry change affects output.

    def process(
        self,
        normalize: Optional[Dict[str, str]] = None,
        aggregateBy: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Performs aggregation and optional normalization based on geographic attributes
        from the geometry object.

        Parameters:
            normalize (Optional[Dict[str, str]]): Dictionary mapping column names to
                                                normalization methods (e.g., {'population': 'area'}).
            aggregateBy (Optional[List[str]]): List of columns (including geographic
                                               attributes like 'county' or 'areatype10',
                                               or index level names) to group by before aggregation.
            mapping (Optional[Dict[str, str]]): Dictionary mapping column names to
                                              aggregation functions (e.g., {'population': 'sum'}).
                                              Required if aggregateBy is specified.

        Returns:
            pd.DataFrame: The processed DataFrame.

        Raises:
             AttributeError: If geometry is required for aggregation/normalization but not set.
             ValueError: If aggregateBy is specified but mapping is None.
             NotImplementedError: If an unknown normalization function is requested.
        """
        print(f"Processing TAZBasedDataFrame ({self.__class__.__name__})...")
        # Accessing dataFrame triggers load/cache
        temp = self.dataFrame
        if temp is None or temp.empty:
            print("DataFrame is empty or None. Skipping process.")
            # Attempt to return an empty DF with correct columns if mapping is provided
            if mapping:
                # Determine potential output columns based on mapping keys
                output_cols = list(mapping.keys())
                if normalize:
                    # Add density columns if normalization requested
                    for col, func in normalize.items():
                        if func == "area":
                            output_cols.append(col + "Density")
                # Need to also consider index levels if aggregateBy is used.
                # Returning a generic empty DF might be safer.
                return pd.DataFrame()
            else:
                return pd.DataFrame()

        if mapping is None and (aggregateBy is not None and len(aggregateBy) > 0):
            raise ValueError(
                "mapping dictionary must be provided if aggregateBy is used."
            )

        outputColumns = set((mapping or dict()).keys())
        additionalColumns = set()
        grouper = None

        # Check if geometry is needed for aggregation/normalization
        geometry_needed = False
        if normalize is not None and any(func == "area" for func in normalize.values()):
            geometry_needed = True
            # Ensure 'gacres' is available if normalizing by area
            # Add 'gacres' to mapping if it's not there and needed for aggregation
            if mapping is not None and "gacres" not in mapping:
                mapping["gacres"] = "sum"  # Assume we want to sum acres for groups
            additionalColumns.add("gacres")

        # Check if geometry attributes are requested for aggregation
        geom_attrs_in_agg = set(aggregateBy or []).intersection(
            {"county", "areatype10", self.geoIndex}
        )
        if geom_attrs_in_agg:
            geometry_needed = True
            # Add geom attributes to additionalColumns if they aren't already output columns or grouper
            additionalColumns.update(
                geom_attrs_in_agg - outputColumns - set(aggregateBy or [])
            )

        # If geometry is needed, merge with gdf BEFORE grouping/aggregation
        if geometry_needed:
            if self.geometry is None or self.geometry.gdf is None:
                raise AttributeError(
                    "You need to define a geometry (with gdf) to perform area normalization or aggregate by geographic attributes."
                )

            # Reset index to perform merge on geoIndex
            temp_reset = temp.reset_index()

            # Ensure the geometry gdf has the geoIndex and the required aggregation/normalization columns ('county', 'areatype10', 'gacres' if needed)
            geom_cols = [self.geometry.index, "county", "areatype10"]
            if "gacres" in additionalColumns:
                geom_cols.append("gacres")
            # Ensure only columns present in gdf are selected
            geom_cols_present = [
                col for col in geom_cols if col in self.geometry.gdf.columns
            ]
            # Ensure geoIndex is always included from gdf for merging
            if self.geometry.index not in geom_cols_present:
                geom_cols_present.append(self.geometry.index)

            # Perform the merge
            # Merge temp_reset (indexed by its original index levels, now columns + original index name column)
            # with geometry gdf (indexed by default integer index, contains geoIndex, county, areatype10, etc.)
            # The merge should happen on the column in temp_reset corresponding to self.geoIndex
            # This column might be the original index name if the original DF was single-indexed,
            # or one of the columns if it was multi-indexed and reset.
            # Need to find the column in temp_reset that corresponds to self.geoIndex.
            # Assume the column in temp_reset that matches the geoIndex name is the one to use.
            # Or, more robustly, if the original DF index contained self.geoIndex as a level name,
            # that column name exists in temp_reset after reset_index().
            # Let's check if self.geoIndex is in the columns of temp_reset.
            if self.geoIndex not in temp_reset.columns:
                # If geoIndex is not a column name after reset_index(), it must have been the original index name
                # when the original DF was single-indexed, or one of the level names in a multi-index.
                # If it was a multi-index, its name becomes a column name after reset_index().
                # If it was a single index with no name, its values become a column named 'index'.
                # This part is tricky and depends on the original DataFrame structure.
                # Let's assume the column name after reset_index() that held the geo-index values
                # has the same name as self.geoIndex, or if the original index had no name,
                # it might be the first column after reset (index 0) or the column named 'index'.
                # A common pattern is for the index level name to become the column name.
                print(
                    f"Warning: Column '{self.geoIndex}' not found directly after reset_index(). Assuming it was the original index name."
                )
                # The original index name should become a column after reset_index().
                # Let's assume the geoIndex column in temp_reset is named the same as self.geoIndex.
                # If the original index was unnamed, this will fail.
                merge_col_left = self.geoIndex  # Assume the column is named the same
                if merge_col_left not in temp_reset.columns:
                    print(
                        f"Error: Could not find column '{self.geoIndex}' or equivalent in reset DataFrame for merging."
                    )
                    # Cannot proceed if merge key is missing
                    return pd.DataFrame()  # Return empty on error

            else:
                merge_col_left = self.geoIndex  # Column name matches geoIndex name

            # Ensure the geometry index is also the merge key on the right side
            merge_col_right = self.geometry.index
            if merge_col_right not in self.geometry.gdf.columns:
                print(
                    f"Error: Geometry GDF does not have column '{merge_col_right}' for merging."
                )
                return pd.DataFrame()  # Return empty on error

            temp_merged = temp_reset.merge(
                self.geometry.gdf[
                    geom_cols_present
                ],  # Select only needed columns from gdf
                left_on=merge_col_left,
                right_on=merge_col_right,
                how="left",  # Keep all rows from temp_reset
                suffixes=(
                    "",
                    "_geom",
                ),  # Add suffix to gdf columns just in case of name conflicts
            )

            # Drop the duplicate merge key column from the right side if it exists and has a suffix
            if (
                f"{merge_col_right}_geom" in temp_merged.columns
                and merge_col_left != merge_col_right
            ):
                temp_merged.drop(columns=[f"{merge_col_right}_geom"], inplace=True)
            # If the merge column names were the same, there's no suffixed column to drop.

            temp = temp_merged  # Use the merged DataFrame for subsequent steps

        # Determine grouper columns
        if aggregateBy is not None and len(aggregateBy) > 0:
            # Ensure all columns in aggregateBy are present in the DataFrame
            missing_agg_cols = [col for col in aggregateBy if col not in temp.columns]
            if missing_agg_cols:
                print(
                    f"Error: Aggregation columns {missing_agg_cols} not found in DataFrame after merging."
                )
                return pd.DataFrame()  # Return empty on error

            grouper = aggregateBy
        else:
            grouper = None  # No grouping

        # Perform aggregation if mapping is provided (and implicitly, if aggregateBy was handled)
        if mapping is not None:
            # Ensure all columns in mapping keys are present in the DataFrame for aggregation
            mapping_cols = list(mapping.keys())
            missing_mapping_cols = [
                col for col in mapping_cols if col not in temp.columns
            ]
            if missing_mapping_cols:
                print(
                    f"Error: Mapping columns {missing_mapping_cols} not found in DataFrame for aggregation."
                )
                return pd.DataFrame()  # Return empty on error

            if grouper is not None:
                # Group and aggregate
                temp_agg = temp.groupby(grouper).agg(mapping)
            else:
                # Aggregate without grouping (aggregate the whole DataFrame)
                # Need to ensure the aggregation functions are valid for the whole DF
                # This path is less common for TAZBasedDataFrame.
                # Let's assume if no grouper, the mapping should apply element-wise or to the whole series.
                # The expected use case for mapping is within aggregation.
                # If no aggregateBy, the mapping is likely intended for the original index.
                # Revisit this case if needed. For now, assume mapping is only used with aggregateBy.
                print(
                    "Warning: Mapping provided without aggregateBy. Applying aggregation to the whole DataFrame."
                )
                # Apply mapping functions to the whole DataFrame columns
                # This might not be what's intended if mapping functions are 'sum', 'mean' etc.
                # Example: {'colA': 'sum', 'colB': 'mean'} on a whole DF doesn't make sense without groupby.
                # Let's perform a sum/mean etc on the whole column.
                agg_results = {}
                for col, func in mapping.items():
                    if callable(func):
                        agg_results[col] = func(temp[col])
                    elif isinstance(func, str):
                        # Use pandas aggregation string
                        agg_results[col] = getattr(
                            temp[col], func
                        )()  # e.g., temp['colA'].sum()
                    else:
                        print(
                            f"Warning: Unknown aggregation function type for column {col}: {func}"
                        )
                        agg_results[col] = np.nan  # Or pd.NA
                temp_agg = pd.DataFrame([agg_results])  # Result is a single row DF
            # Index might need setting depending on desired output format

            temp = temp_agg  # Use the aggregated DataFrame
            print(f"Aggregation complete. Result shape: {temp.shape}")

        # Perform normalization if requested (usually density calculations)
        if normalize is not None:
            # Ensure 'gacres' is available for normalization if needed
            if (
                any(func == "area" for func in normalize.values())
                and "gacres" not in temp.columns
            ):
                # This should not happen if 'gacres' was added to mapping when needed.
                # But check defensively.
                print(
                    "Error: 'gacres' column not available for area normalization after aggregation."
                )
                return pd.DataFrame()  # Return empty on error

            # Ensure columns to normalize exist
            cols_to_normalize = list(normalize.keys())
            missing_norm_cols = [
                col for col in cols_to_normalize if col not in temp.columns
            ]
            if missing_norm_cols:
                print(
                    f"Error: Columns to normalize {missing_norm_cols} not found in DataFrame."
                )
                return pd.DataFrame()  # Return empty on error

            for col, fn in normalize.items():
                if fn == "area":
                    # Create density column
                    temp[col + "Density"] = temp[col].copy() / temp["gacres"]
                    # Add density column to output columns set
                    outputColumns.add(col + "Density")
                    # The original column is also typically kept
                    outputColumns.add(col)
                else:
                    raise NotImplementedError(
                        "Don't have normalization function {0} implemented yet".format(
                            fn
                        )
                    )
            print("Normalization complete.")

        # Select the final output columns based on the set
        # This assumes the index is already set correctly by groupby or is desired.
        # The index names might need careful handling depending on grouper.
        # If mapping was used without grouping, outputColumns might not make sense.
        # Revisit the 'mapping without grouping' case if necessary.

        # For the common case (mapping with groupby), the index is the grouper columns.
        # The columns are the keys from the mapping dictionary, plus any added density columns.
        # The set `outputColumns` correctly contains the keys from `mapping` and the density columns.

        # Ensure all columns in outputColumns are actually in the DataFrame after processing
        final_cols = list(outputColumns.intersection(set(temp.columns)))
        if (
            "gacres" in final_cols and "gacres" not in outputColumns
        ):  # If gacres was added for calculation but not desired in output
            final_cols.remove("gacres")

        return temp[final_cols]

    def toGdf(self):
        """
        Merges the DataFrame with the geometry GeoDataFrame to create a GeoDataFrame.
        Assumes the DataFrame's index contains the geographic zone identifier (self.geoIndex).
        """
        df = self.dataFrame
        if df is None or df.empty:
            print("DataFrame is empty or None. Cannot convert to GeoDataFrame.")
            return gpd.GeoDataFrame()  # Return empty GDF

        if self.geometry is None or self.geometry.gdf is None:
            print("Geometry is not available. Cannot convert to GeoDataFrame.")
            return gpd.GeoDataFrame()

        # The DataFrame's index might be multi-level or single-level.
        # Assume the level corresponding to self.geoIndex exists.
        # Unstack the DataFrame so self.geoIndex becomes a column if it's in a MultiIndex.
        # If it's the single index, its name is self.geoIndex.

        # If the DataFrame is multi-indexed, reset the index and merge on the geoIndex column.
        # If it's single-indexed with name self.geoIndex, just reset index and merge on that column.
        # If it's single-indexed with no name, reset index gives a column 'index'. This case needs care.

        df_to_merge = df.copy()  # Work on a copy
        if isinstance(df_to_merge.index, pd.MultiIndex):
            # Check if geoIndex is one of the levels
            if self.geoIndex not in df_to_merge.index.names:
                print(
                    f"Error: Geo-index level '{self.geoIndex}' not found in MultiIndex."
                )
                return gpd.GeoDataFrame()  # Cannot proceed

            df_to_merge = df_to_merge.reset_index()
            merge_col_left = self.geoIndex  # Column name is the level name
        elif df_to_merge.index.name == self.geoIndex:
            df_to_merge = df_to_merge.reset_index()
            merge_col_left = self.geoIndex  # Column name is the index name
        else:
            print(
                f"Error: DataFrame index name is not '{self.geoIndex}' and it's not a MultiIndex with that level."
            )
            print(
                "Attempting to merge on the first column after reset_index() if index was unnamed."
            )
            df_to_merge = df_to_merge.reset_index()
            # Assume the first column holds the geo-index if the original index was unnamed.
            # This is fragile and might fail.
            if df_to_merge.columns[0] == "index":  # Default name if index had no name
                merge_col_left = "index"
            else:
                print("Error: Could not identify geo-index column after reset_index().")
                return gpd.GeoDataFrame()  # Cannot proceed

        # Ensure the merge column exists in the DataFrame after reset
        if merge_col_left not in df_to_merge.columns:
            print(
                f"Error: Merge column '{merge_col_left}' not found in DataFrame after preparing for merge."
            )
            return gpd.GeoDataFrame()  # Cannot proceed

        # Ensure geometry gdf has the merge column
        merge_col_right = self.geometry.index
        if merge_col_right not in self.geometry.gdf.columns:
            print(
                f"Error: Geometry GDF does not have column '{merge_col_right}' for merging."
            )
            return gpd.GeoDataFrame()  # Cannot proceed

        # Perform the merge
        gdf_merged = self.geometry.gdf.merge(
            df_to_merge,
            left_on=merge_col_right,  # Merge GDF's index column
            right_on=merge_col_left,  # Merge DataFrame's column
            how="left",  # Keep all geometries
            suffixes=("", "_df"),  # Add suffix to DataFrame columns just in case
        )

        # Drop the duplicate merge key column from the right side if it exists and has a suffix
        if (
            f"{merge_col_left}_df" in gdf_merged.columns
            and merge_col_left != merge_col_right
        ):
            gdf_merged.drop(columns=[f"{merge_col_left}_df"], inplace=True)

        return gdf_merged


class NetworkVolumesByLink(OutputDataFrame):
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
            df['VMT_hour'] = df['volume'] * df['length'] / 1609.34  # Assuming length is in meters
            df['VHT_hour_ff'] = df['length'] / df['freespeed'] * df['volume'] / 3600.0

            # Group by link (the first level of the index) and sum VHT across all hours
            # Ensure 'link' is the first level name for groupby
            if df.index.names[0] != "link":
                print(
                    f"Error: Expected 'link' as the first index level name for groupby, but got {df.index.names[0]}."
                )
                return pd.DataFrame(
                    columns=["vht"], index=pd.Index([], name=self.indexedOn)
                )

            df_agg = df.groupby(level="link").agg(vht=("VHT_hour", "sum"), vmt=("VMT_hour", "sum"), vht_ff=("VHT_hour_ff", "sum"))
            df_agg['mph'] = df_agg['vmt'] / df_agg['vht']
            df_agg['mph_ff'] = df_agg['vmt'] / df_agg['vht_ff']
            df_agg['delay_h'] = df_agg['vht'] - df_agg['vht_ff']

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


class NetworkVolumesByLinkByIteration(OutputDataFrame):
    """
    Aggregates NetworkVolumesByLink across multiple iterations.
    Inherits from OutputDataFrame.
    MRO: NetworkVolumesByLinkByIteration -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame
        inputDirectory: BeamRunInputDirectory,  # Needed by OutputDataFrame
        labeledNetwork: LabeledNetwork,  # Dependency, likely for geometry/TAZ mapping downstream
        iterations: List[int],
        inputType: Optional[
            str
        ] = "linkStats",  # Specifies source ('linkStats' or 'pathTraversals')
        *args,  # Accept args/kwargs
        **kwargs,  # Accept args/kwargs
    ):
        # Calculate a hash key based on input directory, class name, iterations, and input type
        m = hashlib.md5()
        input_path_str = str(getattr(inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())
        m.update(str(sorted(iterations)).encode())  # Include sorted iterations in hash
        m.update(inputType.encode())  # Include input type in hash
        override_hash = m.hexdigest()

        # Pass calculated hash_key and other args/kwargs to OutputDataFrame
        super().__init__(
            outputDataDirectory, inputDirectory, hash_key=override_hash, *args, **kwargs
        )

        self.labeledNetwork = labeledNetwork  # Store dependency
        self.iterations = iterations  # Store iterations list
        self.inputType = inputType  # Store input type

        # The final DataFrame will be indexed by link and have columns for each iteration
        self.indexedOn = "link"  # Index is linkId

    def load(self) -> pd.DataFrame:
        """
        Loads and aggregates NetworkVolumesByLink for each specified iteration.
        """
        print(
            f"Loading NetworkVolumesByLinkByIteration for iterations {self.iterations} from {self.inputType} source..."
        )
        temp = dict()
        source_class = None

        if self.inputType.lower() == "linkstats":
            source_class = LinkStatsFile
        elif self.inputType.lower() == "pathtraversals":
            # To use LinkStatsFromPathTraversals, we need the PathTraversalEvents object
            # This requires access to the BeamOutputData instance which has PathTraversalEvents.
            # This class currently only takes a BeamRunInputDirectory.
            # This dependency needs to be resolved.
            # Option 1: Pass BeamOutputData instead of BeamRunInputDirectory.
            # Option 2: Have access to the PathTraversalEvents object via inputDirectory? No.
            # Option 3: Modify LinkStatsFromPathTraversals to take inputDirectory directly? No, it needs processed PTs.
            # Option 4: Only support 'linkStats' input type for now and note the limitation.
            # Let's assume for now we only fully support 'linkStats'.
            print(
                f"Warning: inputType '{self.inputType}' is not fully supported yet. Only 'linkStats' is implemented."
            )
            source_class = LinkStatsFile  # Default to LinkStatsFile even if type is PT

        if source_class:
            for it in sorted(
                self.iterations
            ):  # Process iterations in order for consistent behavior
                print(f"Processing iteration {it}...")
                try:
                    # Get the source object (LinkStatsFile for now)
                    # This requires the inputDirectory object to have a method like linkStatsFile(it)
                    # The BeamRunInputDirectory class *does* have this method.
                    # Accessing inputDirectory from super() should work.
                    source_obj = LinkStatsFromRawFile(
                        self.outputDataDirectory,
                        self.inputDirectory,
                        it,
                    )

                    # Create a NetworkVolumesByLink instance for this iteration's source
                    # Pass self.outputDataDirectory and the specific source object
                    # Pass labeledNetwork as it's a dependency of NetworkVolumesByLink
                    nvbl = NetworkVolumesByLink(
                        self.outputDataDirectory, source_obj, self.labeledNetwork
                    )

                    # Access the dataFrame property which triggers load/preprocess/cache for this iteration's volumes
                    iteration_vht_df = nvbl.dataFrame

                    if iteration_vht_df is not None and not iteration_vht_df.empty:
                        # The result is a DataFrame indexed by 'link' with a 'vht' column
                        temp[f"Iteration {it}"] = iteration_vht_df["vht"]
                        print(
                            f"Successfully loaded VHT for Iteration {it} ({iteration_vht_df.shape[0]} links)."
                        )
                    else:
                        print(f"No VHT data found for Iteration {it}. Skipping.")

                except Exception as e:
                    print(f"Error processing iteration {it}: {e}. Skipping iteration.")
                    # Continue to next iteration

        if temp:
            print("Concatenating VHT data across iterations...")
            # Concatenate the Series into a single DataFrame
            # The index will be 'link' from the source Series/DFs
            combined_df = pd.concat(temp, axis=1)  # Concatenate columns
            combined_df.index.name = self.indexedOn  # Ensure index name is set
            print(
                f"Finished loading NetworkVolumesByLinkByIteration ({combined_df.shape[0]} rows, {combined_df.shape[1]} columns)."
            )
            return combined_df
        else:
            print("No VHT data loaded for any iteration. Returning empty DataFrame.")
            return pd.DataFrame(index=pd.Index([], name=self.indexedOn))

    def preprocess(self, df):
        """
        No specific preprocessing needed after loading and concatenating.
        """
        print(
            f"NetworkVolumesByLinkByIteration preprocess step (no-op). Input shape: {df.shape if df is not None else 'None'}"
        )
        return df


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

        # The inputDirectory for hashing and OutputDataFrame comes from the source object.
        input_dir_from_source = source.inputDirectory

        # Calculate hash key based on the source object's hash and this class name.
        # This ensures the cache location is unique based on the source data (raw or calculated)
        # and the operation performed (labeling).
        m = hashlib.md5()
        # Use input directory from source
        input_path_str = str(getattr(input_dir_from_source, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())  # Use this class's name
        m.update(
            source.hash().encode()
        )  # Hash based on the source's hash (OutputDataFrame's hash)
        calculated_hash = m.hexdigest()

        # Call super() once, passing ALL arguments needed by parents (TAZBasedDataFrame, OutputDataFrame).
        # Match the parameter names expected by the respective __init__ methods.
        # MRO path: LabeledLinkStatsFile -> TAZBasedDataFrame -> OutputDataFrame -> object
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
        # Check if index is a MultiIndex with expected names/levels.
        # Rely on the source's load/preprocess to set index names correctly based on its indexedOn.
        # Here, we just check the names match what we expect based on self.indexedOn.
        if (
            not isinstance(df.index, pd.MultiIndex)
            or list(df.index.names) != self.indexedOn
        ):
            print(
                f"Error: Input LinkStats DataFrame does not have the expected MultiIndex {self.indexedOn}. Actual: {df.index.names}"
            )
            # Attempt to fix the index names if the structure is correct but names are None
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

            # Perform the merge using the 'link' index level of the input df
            # The network_df is indexed by 'linkId', which should match the 'link' level.
            # Ensure the 'link' level is present and named 'link' for `merge` with `left_on`.
            if (
                "link" not in df.index.names
            ):  # Should be handled by index name check above, but defensive
                print("Error: 'link' index level name is missing for merging.")
                return pd.DataFrame(columns=["VMT", "VHT"], index=df.index)

            result_df = mergeLinkstatsWithNetwork(
                df,
                network_df,
                self.geometry.index,  # mergeLinkstatsWithNetwork expects network_df to be indexed by linkId
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

        # Check if required index levels exist (after mergeLinkstatsWithNetwork)
        # The df is indexed by ['link', 'hour']. geoIndex and attributeOrigType are columns.
        # The aggregation needs to happen on columns.
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
        # Accessing self.labeledLinkStatsFile.dataFrame triggers its load/preprocess/cache logic
        df = self.labeledLinkStatsFile.dataFrame
        if df is not None:
            print(
                f"Loaded LabeledLinkStatsFile ({df.shape[0]} rows) for TAZTrafficVolumes."
            )
        else:
            print("Failed to load LabeledLinkStatsFile for TAZTrafficVolumes.")
        return df

    # The process method is inherited from TAZBasedDataFrame and can be used *after*
    # this class's dataFrame is loaded/preprocessed.


class MandatoryLocationsByTaz(TAZBasedDataFrame):
    """
    Represents the count of mandatory locations by TAZ derived from processed persons file.

    Attributes:
        personsFile (ProcessedPersonsFile): The processed persons file.
        geometry (Optional[Geometry]): The geometry object.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame
        personsFile: ProcessedPersonsFile,  # Consumed here as source data
        geometry: Optional[Geometry],  # Needed by TAZBasedDataFrame (passed up)
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):
        # Store the source data object
        self.personsFile = personsFile

        # Pass common arguments (outputDataDirectory, inputDirectory) and TAZBasedDataFrame-specific
        # keyword-only arguments (geometry, geoIndex) to TAZBasedDataFrame parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            personsFile.inputDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            geoIndex=(
                geometry.index if geometry else "TAZ"
            ),  # Keyword-only for TAZBasedDataFrame
            *args,  # Pass any extra positional args
            **kwargs,  # Pass any extra keyword args
        )

        # The index after preprocessing is the geoIndex (e.g., 'TAZ')
        self.indexedOn = self.geoIndex

    def preprocess(self, df):
        """
        Aggregates persons data to count population and jobs by TAZ.

        Parameters:
            df (pd.DataFrame): ProcessedPersonsFile DataFrame.

        Returns:
            pd.DataFrame: DataFrame with population and jobs counts by TAZ.
        """
        print(f"Preprocessing MandatoryLocationsByTaz ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for MandatoryLocationsByTaz is empty or None.")
            # Return empty DataFrame with expected columns/index
            return pd.DataFrame(
                columns=["population", "jobs"], index=pd.Index([], name=self.geoIndex)
            )

        # Ensure required columns exist
        required_cols = [
            "home_zone_id",
            "workplace_zone_id",
            "school_zone_id",
        ]  # TAZ and work_zone_id needed
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found in ProcessedPersonsFile DataFrame for aggregation."
            )
            return pd.DataFrame(
                columns=["population", "jobs", "school_slots"],
                index=pd.Index([], name=self.geoIndex),
            )

        try:
            population = self.countsInColumn(df, "home_zone_id", nonNegative=True)
            workplaces = self.countsInColumn(df, "workplace_zone_id", nonNegative=True)
            schools = self.countsInColumn(df, "school_zone_id", nonNegative=True)

            # Concatenate population and jobs Series into a DataFrame
            # fillna(0) handles TAZs with population but no jobs, or vice versa
            result_df = pd.concat(
                {"population": population, "jobs": workplaces, "school_slots": schools},
                axis=1,
            ).fillna(0)

            # Ensure the index name is set correctly after concatenation
            result_df.index.name = self.geoIndex

            print(
                f"Finished preprocessing MandatoryLocationsByTaz ({result_df.shape[0]} rows)."
            )
            # return result_df.reindex(self.geometry.gdf[self.geoIndex], fill_value=0)
            return result_df.sort_index()
        except Exception as e:
            print(f"Error during MandatoryLocationsByTaz preprocessing: {e}")
            return pd.DataFrame(
                columns=["population", "jobs"], index=pd.Index([], name=self.geoIndex)
            )

    def load(self):
        """
        Loads the processed persons file.
        The aggregation logic is in preprocess().
        """
        print("Loading ProcessedPersonsFile for MandatoryLocationsByTaz...")
        # Accessing self.personsFile.dataFrame triggers its load/preprocess/cache logic
        df = self.personsFile.dataFrame
        if df is not None:
            print(
                f"Loaded ProcessedPersonsFile ({df.shape[0]} rows) for MandatoryLocationsByTaz."
            )
        else:
            print("Failed to load ProcessedPersonsFile for MandatoryLocationsByTaz.")
        return df


class TripModeCount(TAZBasedDataFrame):
    """
    Represents the count of trip modes derived from processed trips file.
    Inherits from TAZBasedDataFrame for spatial processing capabilities.
    MRO: TripModeCount -> TAZBasedDataFrame -> OutputDataFrame -> object

    Attributes:
        tripsFile (ProcessedTripsFile): The processed trips file.
        geometry (Optional[Geometry]): The geometry object (optional, used for spatial processing via TAZBasedDataFrame).
        indexedOn (str or List[str]): The column(s) used as the index for the DataFrame after processing.
        indices (list): List of columns to group by for counting (['trip_mode'] by default).
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame
        tripsFile: ProcessedTripsFile,  # Consumed here as source data
        geometry: Optional[Geometry] = None,  # Needed by TAZBasedDataFrame (passed up)
        # indexedOn: Optional[str] = None, # This is set by __init__ based on indices
        indices: Optional[List[str]] = None,  # Consumed here
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):
        # Store source and specific attributes
        self.tripsFile = tripsFile
        self.indices = indices or ["trip_mode"]  # Store grouping indices

        # Pass common arguments (outputDataDirectory, inputDirectory) and TAZBasedDataFrame-specific
        # keyword-only arguments (geometry, geoIndex) to TAZBasedDataFrame parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            tripsFile.inputDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            *args,  # Pass any extra positional args
            **kwargs,  # Pass any extra keyword args
        )

        # Set the indexedOn property based on the expected index of the processed DataFrame
        # which is determined by the `indices` list.
        self.indexedOn = self.indices if len(self.indices) > 1 else self.indices[0]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Counts trip modes based on the specified grouping indices.

        Parameters:
            df (pd.DataFrame): ProcessedTripsFile DataFrame.

        Returns:
            pd.DataFrame: DataFrame with trip mode counts.
        """
        print(
            f"Preprocessing TripModeCount ({df.shape[0]} rows) with indices {self.indices}..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for TripModeCount is empty or None.")
            # Return empty DataFrame with expected columns/index
            expected_cols = ["count"]
            # Index names are the self.indices list
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        # Define mapping for trip modes (should ideally be a constant)
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
            # Add other relevant modes if necessary
        }

        # Ensure required columns (the self.indices) exist in df
        if not all(col in df.columns for col in self.indices):
            print(
                f"Error: Required columns {self.indices} not found in ProcessedTripsFile DataFrame for counting."
            )
            expected_cols = ["count"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        try:
            # Apply mode mapping to the 'trip_mode' column if it's in the indices
            df_mapped = df.copy()  # Work on a copy to avoid modifying original
            if "trip_mode" in self.indices and "trip_mode" in df_mapped.columns:
                df_mapped["trip_mode"] = df_mapped["trip_mode"].replace(mapping)
                # Handle modes not in mapping - replace with original or 'Other'?
                # The .replace method keeps original if not in mapping by default.

            # Count occurrences based on the specified indices
            # Use dropna=False to include counts of NaN combinations if any
            mode_counts = df_mapped.value_counts(
                self.indices, normalize=False, dropna=False
            )

            # Convert the result Series to a DataFrame and name the count column
            result_df = mode_counts.to_frame("count")

            # The index names are already set by value_counts from self.indices

            print(f"Finished preprocessing TripModeCount ({result_df.shape[0]} rows).")
            return result_df
        except Exception as e:
            print(f"Error during TripModeCount preprocessing: {e}")
            expected_cols = ["count"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

    def load(self):
        """
        Loads the processed trips file.
        The counting and aggregation logic is in preprocess().
        """
        print("Loading ProcessedTripsFile for TripModeCount...")
        # Accessing self.tripsFile.dataFrame triggers its load/preprocess/cache logic
        df = self.tripsFile.dataFrame
        if df is not None:
            print(f"Loaded ProcessedTripsFile ({df.shape[0]} rows) for TripModeCount.")
        else:
            print("Failed to load ProcessedTripsFile for TripModeCount.")
        return df


class TourModeCount(TAZBasedDataFrame):
    """
    Represents the count of tour modes derived from processed tours file.
    This class provides functionality to load and preprocess the count of tour modes obtained from processed tours data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        toursFile (ProcessedToursFile): The processed tours file.
        geometry (Optional[Geometry]): The geometry object (optional, used for spatial processing via TAZBasedDataFrame).
        indexedOn (str): The column used as the index for the DataFrame ('tour_mode' by default).
        indices (list): List of columns to group by for counting (['tour_mode'] by default).

    Methods:
        load(): Loads the count of tour modes from the processed tours file.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        toursFile: ProcessedToursFile,
        geometry: Optional[Geometry] = None,
        # indexedOn: Optional[str] = None, # This is set by __init__ based on indices
        indices: Optional[List[str]] = None,  # Consumed here
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):

        # Store source and specific attributes
        self.toursFile = toursFile
        self.indices = indices or ["tour_mode"]  # Store grouping indices

        # Pass common arguments (outputDataDirectory, inputDirectory) and TAZBasedDataFrame-specific
        # keyword-only arguments (geometry, geoIndex) to TAZBasedDataFrame parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            toursFile.inputDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            geoIndex=(
                geometry.index if geometry else "TAZ"
            ),  # Keyword-only for TAZBasedDataFrame
            *args,  # Pass any extra positional args
            **kwargs,  # Pass any extra keyword args
        )

        # Set the indexedOn property based on the expected index of the processed DataFrame
        # which is determined by the `indices` list.
        self.indexedOn = self.indices if len(self.indices) > 1 else self.indices[0]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        print(
            f"Preprocessing TourModeCount ({df.shape[0]} rows if not None) with indices {self.indices}..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for TourModeCount is empty or None.")
            # Return empty DataFrame with expected columns/index
            expected_cols = ["count"]
            # Index names are the self.indices list
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        # Define mapping for tour modes (should ideally be a constant, same as trip modes?)
        # Assuming tour mode mapping is the same as trip mode mapping
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
            # Add other relevant modes if necessary
        }

        # Ensure required columns (the self.indices) exist in df
        if not all(col in df.columns for col in self.indices):
            print(
                f"Error: Required columns {self.indices} not found in ProcessedToursFile DataFrame for counting."
            )
            expected_cols = ["count"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        try:
            # Apply mode mapping to the 'tour_mode' column if it's in the indices
            df_mapped = df.copy()  # Work on a copy to avoid modifying original
            if "tour_mode" in self.indices and "tour_mode" in df_mapped.columns:
                df_mapped["tour_mode"] = df_mapped["tour_mode"].replace(mapping)
                # Handle modes not in mapping - replace with original or 'Other'?

            # Count occurrences based on the specified indices
            # Use dropna=False to include counts of NaN combinations if any
            mode_counts = df_mapped.value_counts(
                self.indices, normalize=False, dropna=False
            )

            # Convert the result Series to a DataFrame and name the count column
            result_df = mode_counts.to_frame("count")

            # The index names are already set by value_counts from self.indices

            print(f"Finished preprocessing TourModeCount ({result_df.shape[0]} rows).")
            return result_df

        except (
            KeyError,
            AttributeError,
            Exception,
        ) as e:  # Catch potential errors like column not found or replace failure
            print(
                f"Error during TourModeCount preprocessing: {e}. Skipping processing."
            )
            expected_cols = ["count"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

    def load(self):

        print("Loading ProcessedToursFile for TourModeCount...")
        # Accessing self.toursFile.dataFrame triggers its load/preprocess/cache logic
        df = self.toursFile.dataFrame
        if df is not None:
            print(f"Loaded ProcessedToursFile ({df.shape[0]} rows) for TourModeCount.")
        else:
            print("Failed to load ProcessedToursFile for TourModeCount.")
        return df


class TripPMT(OutputDataFrame):
    """
    Represents the person miles traveled (PMT) for each trip mode derived from processed trips and skims files.

    This class provides functionality to load and preprocess the person miles traveled (PMT) for each trip mode obtained from processed trips and skims data.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        tripsFile (ProcessedTripsFile): The processed trips file.
        skimsFile (ProcessedSkimsFile): The processed skims file.
        indexedOn (str): The column used as the index for the DataFrame ('trip_mode' by default).
        indices (list): List of columns to group by for calculating PMT (['trip_mode'] by default).

    Methods:
        load(): Loads the person miles traveled (PMT) for each trip mode from the processed trips and skims files.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame
        tripsFile: ProcessedTripsFile,  # Consumed here as source data
        skimsFile: ProcessedSkimsFile,  # Consumed here as source data
        # indexedOn: Optional[str] = None, # This is set by __init__ based on indices
        indices: Optional[List[str]] = None,  # Consumed here
        *args,  # Accept any extra args
        **kwargs,  # Accept any extra kwargs
    ):
        # Store sources and specific attributes
        self.tripsFile = tripsFile
        self.skimsFile = skimsFile
        self.indices = indices or ["trip_mode"]  # Store grouping indices

        # Pass common arguments (outputDataDirectory, inputDirectory) to OutputDataFrame parent.
        super().__init__(outputDataDirectory, tripsFile.inputDirectory, *args, **kwargs)

        # Set the indexedOn property based on the expected index of the processed DataFrame
        # which is determined by the `indices` list.
        self.indexedOn = self.indices if len(self.indices) > 1 else self.indices[0]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates PMT for trips using skim distances and specified grouping indices.

        Parameters:
            df (pd.DataFrame): ProcessedTripsFile DataFrame with distanceInMiles column added.

        Returns:
            pd.DataFrame: DataFrame with PMT totals.
        """
        print(
            f"Preprocessing TripPMT ({df.shape[0]} rows) with indices {self.indices}..."
        )
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for TripPMT is empty or None.")
            # Return empty DataFrame with expected columns/index
            expected_cols = ["distanceInMiles"]
            # Index names are the self.indices list
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        # Ensure required columns exist
        required_cols = self.indices + ["distanceInMiles"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found in DataFrame for PMT aggregation."
            )
            expected_cols = ["distanceInMiles"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

        # Define mapping for trip modes (should ideally be a constant)
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
            # Add other relevant modes if necessary
        }

        try:
            # Apply mode mapping to the 'trip_mode' column if it's in the indices
            df_mapped = df.copy()  # Work on a copy to avoid modifying original
            if "trip_mode" in self.indices and "trip_mode" in df_mapped.columns:
                df_mapped["trip_mode"] = df_mapped["trip_mode"].replace(mapping)

            # Ensure distanceInMiles is numeric
            df_mapped["distanceInMiles"] = pd.to_numeric(
                df_mapped["distanceInMiles"], errors="coerce"
            ).fillna(0)

            # Group by the specified indices and sum the distanceInMiles
            result_df = df_mapped.groupby(self.indices).agg({"distanceInMiles": "sum"})

            # The index names are already set by groupby from self.indices

            print(f"Finished preprocessing TripPMT ({result_df.shape[0]} rows).")
            return result_df
        except Exception as e:
            print(f"Error during TripPMT preprocessing: {e}")
            expected_cols = ["distanceInMiles"]
            return pd.DataFrame(
                columns=expected_cols,
                index=pd.MultiIndex.from_tuples([], names=self.indices),
            )

    def load(self):
        """
        Loads the processed trips file and adds skim distances.
        The aggregation logic is in preprocess().
        """
        print("Loading ProcessedTripsFile and SkimsFile for TripPMT...")
        # Accessing dataFrame/file triggers load/cache for source data
        trips = self.tripsFile.dataFrame
        skims = self.skimsFile.dataFrame

        if trips is None or trips.empty:
            print("ProcessedTripsFile is empty or None. Cannot load for TripPMT.")
            return None  # Cannot proceed without trips data

        if skims is None or skims.empty:
            print(
                "ProcessedSkimsFile is empty or None. Cannot add distances for TripPMT."
            )
            # Decide how to handle: return trips DF without distance? Return None?
            # Let's return None as PMT cannot be calculated without distance.
            return None

        # Ensure required columns exist in trips
        required_trips_cols = ["origin", "destination"]
        if not all(col in trips.columns for col in required_trips_cols):
            print(
                f"Error: Required trip columns {required_trips_cols} not found for merging skims."
            )
            return None

        # Ensure required index exists in skims
        required_skims_index = ["Origin", "Destination"]
        if (
            not isinstance(skims.index, pd.MultiIndex)
            or list(skims.index.names) != required_skims_index
        ):
            print(
                f"Error: Skims DataFrame does not have the expected index {required_skims_index}."
            )
            return None

        try:
            # Add distanceInMiles column to the trips DataFrame
            # Reindex the skims DataFrame by creating a MultiIndex from trips' origin/destination columns
            # Ensure origin and destination columns in trips are of compatible type with skims index levels
            trips["origin"] = pd.to_numeric(trips["origin"], errors="coerce")
            trips["destination"] = pd.to_numeric(trips["destination"], errors="coerce")

            # Drop trips where origin or destination could not be converted to numeric (NaN)
            trips_valid_loc = trips.dropna(subset=["origin", "destination"]).copy()

            # Ensure skims has the 'DistanceMiles' column
            if "DistanceMiles" not in skims.columns:
                print(
                    "Error: Skims DataFrame does not have the 'DistanceMiles' column."
                )
                return None

            # Create MultiIndex from valid trips origins/destinations
            trips_origin_dest_idx = pd.MultiIndex.from_frame(
                trips_valid_loc[["origin", "destination"]], names=required_skims_index
            )

            # Reindex skims to match the trips' origin/destination pairs
            # This aligns skim distances to each trip based on its O-D pair
            # The result is a Series indexed by the trips_origin_dest_idx
            distance_series = skims["DistanceMiles"].reindex(trips_origin_dest_idx)

            # Assign the distance Series back to the DataFrame
            # Aligning by index ensures distances go to the correct original trip rows
            # Need to align based on the index *before* dropping NA, then join?
            # Let's reindex the distance series to the index of trips_valid_loc
            distance_series.index = (
                trips_valid_loc.index
            )  # Align index with the valid trips DF

            trips_valid_loc["distanceInMiles"] = distance_series

            print(
                f"Loaded trips and added skim distances ({trips_valid_loc.shape[0]} rows)."
            )
            return trips_valid_loc  # Return the trips DF with the distance column
        except Exception as e:
            print(f"Error during TripPMT load (merging skims): {e}")
            return None  # Indicate loading failure


class MeanDistanceToWork(TAZBasedDataFrame):  # Inherit from TAZBasedDataFrame
    """
    Calculates the mean distance to work by TAZ using persons and skims files.
    Inherits from TAZBasedDataFrame as its output is spatially indexed by TAZ.
    MRO: MeanDistanceToWork -> TAZBasedDataFrame -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame
        personsFile: ProcessedPersonsFile,  # Consumed here as source
        skimsFile: ProcessedSkimsFile,  # Consumed here as source
        geometry: Optional[Geometry] = None,  # Needed by TAZBasedDataFrame (passed up)
        *args,  # Accept extra args
        **kwargs,  # Accept extra kwargs
    ):
        # Store sources
        self.personsFile = personsFile
        self.skimsFile = skimsFile

        # The output index is TAZ. Set this as geoIndex and indexedOn after preprocessing.
        # Pass common arguments (outputDataDirectory, inputDirectory) and TAZBasedDataFrame-specific
        # keyword-only arguments (geometry, geoIndex) to TAZBasedDataFrame parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            personsFile.inputDirectory,  # Positional for OutputDataFrame/TAZBasedDataFrame
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            geoIndex=(
                geometry.index if geometry else "TAZ"
            ),  # Keyword-only for TAZBasedDataFrame
            *args,  # Pass any extra positional args
            **kwargs,  # Pass any extra keyword args
        )

        # Set the indexedOn property based on the expected index of the processed DataFrame
        self.indexedOn = self.geoIndex  # Output is indexed by geoIndex (e.g., 'TAZ')

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean distance to work by TAZ.

        Parameters:
            df (pd.DataFrame): DataFrame containing persons filtered for workers with skim distances added.

        Returns:
            pd.DataFrame: DataFrame with mean distance to work by TAZ.
        """
        print(f"Preprocessing MeanDistanceToWork ({df.shape[0]} rows)...")
        # Ensure df is not None or empty before processing
        if df is None or df.empty:
            print("Input DataFrame for MeanDistanceToWork is empty or None.")
            # Return empty DataFrame with expected columns/index
            return pd.DataFrame(
                columns=["meanDistance"],
                index=pd.Index([], name=self.geoIndex),  # Index by geoIndex
            )

        # Ensure required columns exist in df (TAZ, work_zone_id, distanceInMiles)
        required_cols = [self.geoIndex, "work_zone_id", "distanceInMiles"]
        if not all(col in df.columns for col in required_cols):
            print(
                f"Error: Required columns {required_cols} not found in DataFrame for aggregation."
            )
            return pd.DataFrame(
                columns=["meanDistance"], index=pd.Index([], name=self.geoIndex)
            )

        try:
            # Group by TAZ (using self.geoIndex) and aggregate distanceInMiles by sum and count (size)
            # Ensure distanceInMiles is numeric
            df["distanceInMiles"] = pd.to_numeric(
                df["distanceInMiles"], errors="coerce"
            ).fillna(0)

            # Grouping by self.geoIndex which is a column in the input df for preprocess
            # The input df for preprocess is the result of load(), which adds 'distanceInMiles'
            # to the filtered persons DF (indexed by person_id).
            # The input DF to preprocess should have 'TAZ', 'work_zone_id', 'distanceInMiles' as columns.
            # The grouping should be by 'TAZ' (or self.geoIndex).
            byTaz = df.groupby(self.geoIndex).agg(
                {
                    "distanceInMiles": ["sum", "size"]
                }  # Use size for count, it includes NaNs if not dropped
                # but we dropped NaNs in load, so size is count of valid rows.
            )

            # The aggregation result is a DataFrame with MultiIndex columns ('distanceInMiles', ['sum', 'size'])
            byTaz.columns = byTaz.columns.droplevel(
                0
            )  # Drop the 'distanceInMiles' level from columns

            # Calculate meanDistance, handling division by zero size
            byTaz["meanDistance"] = byTaz["sum"] / byTaz["size"].replace(
                0, np.nan
            )  # Replace 0 size with NaN

            # Select the 'meanDistance' column and convert to a DataFrame
            result_df = byTaz["meanDistance"].to_frame()

            # The index name is already set by groupby

            print(
                f"Finished preprocessing MeanDistanceToWork ({result_df.shape[0]} rows)."
            )
            return result_df
        except Exception as e:
            print(f"Error during MeanDistanceToWork preprocessing: {e}")
            return pd.DataFrame(
                columns=["meanDistance"], index=pd.Index([], name=self.geoIndex)
            )

    def load(self):
        """
        Loads processed persons and skims, filters persons for workers, and adds skim distances.
        The aggregation logic is in preprocess().
        """
        print("Loading ProcessedPersonsFile and SkimsFile for MeanDistanceToWork...")
        # Accessing dataFrame/file triggers load/cache for source data
        persons = self.personsFile.dataFrame
        skims = self.skimsFile.dataFrame

        if persons is None or persons.empty:
            print(
                "ProcessedPersonsFile is empty or None. Cannot load for MeanDistanceToWork."
            )
            return None  # Cannot proceed without persons data

        if skims is None or skims.empty:
            print(
                "ProcessedSkimsFile is empty or None. Cannot add distances for MeanDistanceToWork."
            )
            return None  # Cannot proceed without skims data

        # Filter persons for workers with a valid work_zone_id (> 0)
        # Ensure work_zone_id is numeric
        persons["work_zone_id"] = pd.to_numeric(
            persons["work_zone_id"], errors="coerce"
        )
        workers_df = persons.loc[
            persons["work_zone_id"] > 0, [self.geoIndex, "work_zone_id"]
        ].copy()  # Select TAZ and work_zone_id, and make a copy

        if workers_df.empty:
            print(
                "No workers found with valid work_zone_id. Returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=[self.geoIndex, "work_zone_id", "distanceInMiles"]
            )

        # Ensure required skims index exists
        required_skims_index = ["Origin", "Destination"]
        if (
            not isinstance(skims.index, pd.MultiIndex)
            or list(skims.index.names) != required_skims_index
        ):
            print(
                f"Error: Skims DataFrame does not have the expected index {required_skims_index}."
            )
            return None

        # Ensure skims has the 'DistanceMiles' column
        if "DistanceMiles" not in skims.columns:
            print("Error: Skims DataFrame does not have the 'DistanceMiles' column.")
            return None

        try:
            # Add distanceInMiles column using skims
            # Create MultiIndex from workers_df's TAZ (origin) and work_zone_id (destination) columns
            # Rename columns temporarily to match skims index names
            workers_origin_dest_idx = pd.MultiIndex.from_frame(
                workers_df.rename(
                    columns={self.geoIndex: "Origin", "work_zone_id": "Destination"}
                ),
                names=required_skims_index,
            )

            # Reindex skims to match the worker's O-D pairs (Home TAZ to Work TAZ)
            # This aligns skim distances to each worker based on their home-to-work pair
            # The result is a Series indexed by the workers_origin_dest_idx
            distance_series = skims["DistanceMiles"].reindex(workers_origin_dest_idx)

            # Assign the distance Series back to the DataFrame, aligning by index (person_id)
            # Need to align based on the index of the original workers_df
            distance_series.index = workers_df.index  # Align index with the workers_df

            workers_df["distanceInMiles"] = distance_series

            print(
                f"Loaded workers and added skim distances ({workers_df.shape[0]} rows)."
            )
            return workers_df  # Return the workers DF with the distance column
        except Exception as e:
            print(f"Error during MeanDistanceToWork load (merging skims): {e}")
            return None  # Indicate loading failure


class TripModeCountByOrigin(TripModeCount):
    """
    Represents the count of trip modes by origin derived from processed trips file.
    Inherits from TripModeCount (for mode counting logic) and TAZBasedDataFrame (implicitly via TripModeCount).
    Sets the grouping indices to ['trip_mode', 'origin'].
    MRO: TripModeCountByOrigin -> TripModeCount -> TAZBasedDataFrame -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        tripsFile: ProcessedTripsFile,
        geometry: Optional[Geometry] = None,
        *args,  # Accept extra args
        **kwargs,  # Accept extra kwargs
    ):
        # Set the grouping indices for the parent TripModeCount class
        indices_for_parent = ["trip_mode", "origin"]
        # The geoIndex for TAZBasedDataFrame logic should be 'origin' for this class
        geoIndex_for_parent = (
            "origin"  # Use origin column as geo index for spatial processing if needed
        )

        # Pass all args up to TripModeCount, which will pass common args
        # and TAZBasedDataFrame args up its chain.
        super().__init__(
            outputDataDirectory,  # Positional for TripModeCount / TAZBasedDataFrame / OutputDataFrame
            tripsFile,  # Positional for TripModeCount
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame (passed by TripModeCount's super)
            geoIndex=geoIndex_for_parent,  # Keyword-only for TAZBasedDataFrame (passed by TripModeCount's super)
            indices=indices_for_parent,  # Keyword-only for TripModeCount (consumed by TripModeCount)
            *args,  # Pass extra args
            **kwargs,  # Pass extra kwargs
        )

        # The indexedOn property is already set correctly in the TripModeCount __init__
        # based on the 'indices' argument passed above.
        # self.indexedOn = self.indices # Redundant


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
        # Set the grouping indices for the parent TripModeCount class
        indices_for_parent = ["trip_mode", "primary_purpose"]

        # Pass all args up to TripModeCount, which will pass common args
        # and TAZBasedDataFrame args up its chain.
        # Note: This class does NOT need geometry, so we don't pass it.
        # TripModeCount defaults geometry to None, so TAZBasedDataFrame will be initialized without geometry.
        super().__init__(
            outputDataDirectory,  # Positional for TripModeCount / TAZBasedDataFrame / OutputDataFrame
            tripsFile,  # Positional for TripModeCount
            indices=indices_for_parent,  # Keyword-only for TripModeCount (consumed by TripModeCount)
            # geometry=None, # TAZBasedDataFrame needs geometry keyword, but defaults to None
            # geoIndex=None, # TAZBasedDataFrame needs geoIndex keyword, but defaults to "TAZ"
            # We rely on TripModeCount passing None for geometry and default for geoIndex
        )

        # The indexedOn property is already set correctly in the TripModeCount __init__
        # based on the 'indices' argument passed above.
        # self.indexedOn = self.indices # Redundant


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
        # Set the grouping indices for the parent TripPMT class
        indices_for_parent = ["trip_mode", "origin"]

        # Pass all args up to TripPMT, which will pass common args
        # up its chain.
        super().__init__(
            outputDataDirectory,  # Positional for TripPMT / OutputDataFrame
            tripsFile,  # Positional for TripPMT
            skimsFile,  # Positional for TripPMT
            indices=indices_for_parent,  # Keyword-only for TripPMT (consumed by TripPMT)
        )

        # The indexedOn property is already set correctly in the TripPMT __init__
        # based on the 'indices' argument passed above.
        # self.indexedOn = self.indices # Redundant

        # This class also needs the geoIndex attribute for the accessor in TripPMTByCountyByYear
        self.geoIndex = "origin"  # Use origin column as geo index


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
        # Set the grouping indices for the parent TripPMT class
        indices_for_parent = ["trip_mode", "primary_purpose"]

        # Pass all args up to TripPMT, which will pass common args
        # up its chain.
        super().__init__(
            outputDataDirectory,  # Positional for TripPMT / OutputDataFrame
            tripsFile,  # Positional for TripPMT
            skimsFile,  # Positional for TripPMT
            indices=indices_for_parent,  # Keyword-only for TripPMT (consumed by TripPMT)
        )

        # The indexedOn property is already set correctly in the TripPMT __init__
        # based on the 'indices' argument passed above.
        # self.indexedOn = self.indices # Redundant


class InfoByYear(OutputDataFrame):
    """

    Base class for output dataframes that aggregate data across years,

    typically using the last iteration for each year.

    Inherits from OutputDataFrame.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (InputDirectory): The input directory (PilatesRunInputDirectory).
        pilatesInputDict (Dict[Tuple[int, int], "ModelOutputData"]): Dict of run data.
        accessor (Callable): Function to access data from a ModelOutputData instance.
        columns (List[str]): Expected index names of the accessor's output DataFrame.
        __lastIterationPerYear (Dict[int, int]): Mapping from year to the last iteration found.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Positional for OutputDataFrame
        inputDirectory: InputDirectory,  # Positional for OutputDataFrame (expected to be PilatesRunInputDirectory)
        pilatesInputDict: Dict[Tuple[int, int], "ModelOutputData"],
        accessor: Callable[["ModelOutputData"], Optional[pd.DataFrame]],
        columns: List[str],  # Expected index names
        *args,
        **kwargs,
    ):

        # We need to determine the last iteration first for hashing.
        # Do this BEFORE calculating the hash key.
        last_iteration_per_year = {}
        # Ensure pilatesInputDict is not None or empty before iterating
        if pilatesInputDict:
            for yr, it in pilatesInputDict.keys():
                if (
                    yr not in last_iteration_per_year
                    or it >= last_iteration_per_year[yr]
                ):
                    last_iteration_per_year[yr] = it
        self.__lastIterationPerYear = last_iteration_per_year  # Store for load method

        # Calculate hash key based on input directory and last iteration per year
        m = hashlib.md5()
        input_path_str = str(getattr(inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())
        m.update(
            str(sorted(self.__lastIterationPerYear.items())).encode()
        )  # Include years and their LAST iteration
        override_hash = m.hexdigest()

        # Pass common arguments (outputDataDirectory, inputDirectory) and the
        # calculated hash_key to the OutputDataFrame parent, along with any
        # additional *args and **kwargs.
        super().__init__(
            outputDataDirectory,  # Positional argument for OutputDataFrame
            inputDirectory,  # Positional argument for OutputDataFrame
            hash_key=override_hash,  # Keyword-only argument for OutputDataFrame
            *args,
            **kwargs,
        )

        # Store InfoByYear specific attributes after calling super()
        self.pilatesInputDict = pilatesInputDict
        self.__accessor = accessor
        self.__columns = columns  # Store expected index names from accessor's DF

        # _diskLocation is already set by OutputDataFrame using the hash_key

    # The hash method in OutputDataFrame is now updated to use _hash_key if provided.
    # No need to override hash() here explicitly if passing hash_key to super.
    # def hash(self):
    #      return self._override_hash

    def load(self):
        """
        Loads data from the last available iteration for each year using the accessor.
        Concatenates the results.
        """

        print(
            f"Loading data for {self.__class__.__name__} (last iteration per year)..."
        )
        data_frames = {}

        # Iterate through years and their determined last iteration, sorted by year
        # Use the stored lastIterationPerYear dictionary
        if self.__lastIterationPerYear:
            for yr in sorted(self.__lastIterationPerYear.keys()):
                it = self.__lastIterationPerYear[yr]
                print(f" - Attempting to load year {yr}, last iteration {it}")
                # Try to load the data, iterate backward through iterations if the last one fails
                current_it = it
                df = None

                # Check down to iteration -1, assuming -1 is a valid key if it exists in the dict
                # Check iterations from 'it' down to -1.
                # Need to find all iterations for this year first.
                iterations_for_year = sorted(
                    [i for (y, i) in self.pilatesInputDict.keys() if y == yr],
                    reverse=True,
                )

                # Find the index of the target 'it' in the sorted iterations for the year
                try:
                    start_idx = iterations_for_year.index(it)
                except ValueError:
                    print(
                        f"Warning: Last iteration {it} for year {yr} not found in pilatesInputDict keys. Skipping year."
                    )
                    continue  # Skip this year if the last iteration is not in the dictionary

                for i in range(start_idx, len(iterations_for_year)):
                    current_it = iterations_for_year[i]
                    print(f" - Attempting to load year {yr}, iteration {current_it}")
                    try:
                        if (yr, current_it) in self.pilatesInputDict:
                            data_instance = self.pilatesInputDict[(yr, current_it)]

                            # Access the data using the provided accessor
                            df = self.__accessor(data_instance)

                            if df is not None and not df.empty:
                                print(
                                    f" - Successfully loaded year {yr}, iteration {current_it} ({df.shape[0]} rows)"
                                )
                                data_frames[yr] = df
                                break  # Found data for this year, move to next year

                            else:
                                print(
                                    f" - Accessor returned no data for year {yr}, iteration {current_it}. Trying previous iteration."
                                )
                        else:
                            # This case should be handled by iterating over iterations_for_year
                            print(
                                f" - Run data for year {yr}, iteration {current_it} not found unexpectedly. Skipping."
                            )  # Should not happen within sorted iterations_for_year

                    except Exception as e:
                        print(
                            f" - Error loading data for year {yr}, iteration {current_it}: {e}. Trying previous iteration."
                        )
                        raise e
                        # Optionally log the error more verbosely

            if (
                yr not in data_frames
            ):  # Check if data was successfully loaded for the year after trying iterations
                print(
                    f" - Could not load data for year {yr} after trying all iterations down to {iterations_for_year[-1]}."
                )

        if not data_frames:

            print(f"No data found across any year for {self.__class__.__name__}.")

            # Determine expected index names from the 'columns' attribute
            # If the accessor returns a Series, self.__columns might be the Series name like ['count']
            # The concatenated DF should have index [year] + self.__columns
            concat_names_if_empty = ["year"] + self.__columns

            # Return empty DF with expected index names
            return pd.DataFrame(columns=[]).set_index(
                pd.MultiIndex.from_tuples([], names=concat_names_if_empty)
            )

        # Concatenate DataFrames. The keys of data_frames dict are years.

        # Use index names including 'year' plus the accessor's DF index names

        # Get the index names from the first loaded dataframe if available
        # Accessing the index of the first DF handles cases where accessor returns Series or DF
        first_df_index = next(iter(data_frames.values())).index
        first_df_index_names = list(
            first_df_index.names or []
        )  # Use list() to handle None

        # Filter out None from index names before concatenating
        valid_index_names = [name for name in first_df_index_names if name is not None]

        concat_names = ["year"] + valid_index_names

        # Handle cases where the accessor returns a Series (index.names is [None] or just None)
        # In this case, the Series name becomes a column when converted to DF for concat.
        # The desired concatenated index is usually [year, SeriesName].
        if not valid_index_names and isinstance(
            next(iter(data_frames.values())), pd.Series
        ):
            # If Series, convert to DataFrame, column name defaults to Series name
            # The index name becomes None. We need to use the Series name for the concat index.
            series_name = next(iter(data_frames.values())).name
            if series_name is None:
                series_name = "value"  # Default name if series has no name
            concat_names = ["year", series_name]

            # Convert Series to DataFrame for concatenation consistency
            data_frames = {
                k: v.to_frame(name=series_name) for k, v in data_frames.items()
            }  # Convert Series to DF with a column name
        elif not valid_index_names and isinstance(
            next(iter(data_frames.values())), pd.DataFrame
        ):
            # If DF with unnamed index, the columns should be used? Or the default '0', '1', ... column names?
            # This case is less clear. Assuming the accessor returns a meaningful DF index or Series.
            # If the DF is indexed by default integers (index.name is None), the original columns
            # of that DF should be preserved. Concat names become [year] + original column names.
            # Check if the first DF has a default integer index and no index name.
            first_df = next(iter(data_frames.values()))
            if first_df.index.name is None and isinstance(
                first_df.index, pd.RangeIndex
            ):
                print(
                    f"Warning: Accessor returned DataFrame with default integer index for {self.__class__.__name__}. Concatenated columns will include original DF columns."
                )
                # Concat names will be [year] + original column names
                # pd.concat handles column alignment automatically.
                pass  # Keep concat_names as ["year"] + [] = ["year"] for now, columns will be added by concat

        print(f"Concatenating dataframes with names: {concat_names}")

        combined_df = pd.concat(data_frames, axis=0)  # Concatenate rows

        # Set the indexedOn attribute based on the concatenated DataFrame's index names
        # Ensure index has names before converting to list
        self.indexedOn = (
            list(combined_df.index.names) if combined_df.index.names else []
        )

        print(
            f"Finished loading {self.__class__.__name__}. Result shape: {combined_df.shape}"
        )

        return combined_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Base class preprocess does nothing by default
        return df


class TripPMTByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        # Define the accessor function to get the data for a single year/iteration
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access the tripPMT dataframe. Its index names are set by TripPMT.__init__.
            # By default, TripPMT has indices ['trip_mode'].
            # So the accessor returns a DataFrame indexed by 'trip_mode'.
            return outputData.tripPMT.dataFrame

        # The index name of the accessor's output DF is 'trip_mode'.
        columns = ["trip_mode"]  # Expected index names of the accessor's output

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # Attributes like __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear

    def load(self):
        """
        Loads data using the InfoByYear base class logic.
        The accessor gets the TripPMT data for the last iteration of each year.
        InfoByYear's load concatenates these.
        """
        # The load method is inherited from InfoByYear and handles collecting and concatenating data across years.
        # Accessing self.dataFrame will trigger InfoByYear.load().
        return super().load()  # Explicitly call the parent load method


class TripPMTByPrimaryPurposeByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        # Define the accessor function to get the data for a single year/iteration
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access the tripPMTByPrimaryPurpose dataframe. Its index names are set by TripPMTByPrimaryPurpose.__init__.
            # TripPMTByPrimaryPurpose has indices ['trip_mode', 'primary_purpose'].
            # So the accessor returns a DataFrame indexed by ['trip_mode', 'primary_purpose'].
            return outputData.tripPMTByPrimaryPurpose.dataFrame

        # The index names of the accessor's output DF are ['trip_mode', 'primary_purpose'].
        columns = [
            "trip_mode",
            "primary_purpose",
        ]  # Expected index names of the accessor's output

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # Attributes like __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear

    def load(self):
        """
        Loads data using the InfoByYear base class logic.
        The accessor gets the TripPMTByPrimaryPurpose data for the last iteration of each year.
        InfoByYear's load concatenates these.
        """
        # The load method is inherited from InfoByYear and handles collecting and concatenating data across years.
        # Accessing self.dataFrame will trigger InfoByYear.load().
        return super().load()  # Explicitly call the parent load method


class InfoByIteration(OutputDataFrame):
    """

    Base class for output dataframes that aggregate data across iterations.

    Loads data from *all* available iterations for the specified year.

    Inherits from OutputDataFrame.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (InputDirectory): The input directory (PilatesRunInputDirectory).
        pilatesInputDict (Dict[Tuple[int, int], "ModelOutputData"]): Dict of run data.
        accessor (Callable): Function to access data from a ModelOutputData instance.
        columns (List[str]): Expected index names of the accessor's DataFrame index.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Positional for OutputDataFrame
        inputDirectory: InputDirectory,  # Positional for OutputDataFrame (expected to be PilatesRunInputDirectory)
        pilatesInputDict: Dict[Tuple[int, int], "ModelOutputData"],
        accessor: Callable[["ModelOutputData"], Optional[pd.DataFrame]],
        columns: List[str],
        *args,
        **kwargs,
    ):

        # Calculate hash key based on input directory and all iterations
        m = hashlib.md5()
        input_path_str = str(getattr(inputDirectory, "directoryPath", ""))
        m.update(input_path_str.encode())
        m.update(self.__class__.__name__.encode())
        m.update(
            str(sorted(pilatesInputDict.keys())).encode()
        )  # Include all years/iters
        override_hash = m.hexdigest()

        # Pass common arguments (outputDataDirectory, inputDirectory) and the
        # calculated hash_key to the OutputDataFrame parent, along with any
        # additional *args and **kwargs.
        super().__init__(
            outputDataDirectory,  # Positional argument for OutputDataFrame
            inputDirectory,  # Positional argument for OutputDataFrame
            hash_key=override_hash,  # Keyword-only argument for OutputDataFrame
            *args,
            **kwargs,
        )

        # Store InfoByIteration specific attributes after calling super()
        self.pilatesInputDict = pilatesInputDict
        self.__accessor = accessor
        self.__columns = columns  # Store expected index names

        # _diskLocation is already set by OutputDataFrame using the hash_key

    # The hash method in OutputDataFrame is now updated to use _hash_key if provided.
    # No need to override hash() here explicitly if passing hash_key to super.
    # def hash(self):
    #      return self._override_hash

    def load(self):
        """
        Loads data from all available iterations for each year using the accessor.
        Concatenates the results.
        """
        print(f"Loading data for {self.__class__.__name__} across all iterations...")
        data_frames = {}

        # Sort keys to ensure consistent order for concatenation
        # Ensure pilatesInputDict is not None or empty before iterating
        if self.pilatesInputDict:
            for yr, it in sorted(self.pilatesInputDict.keys()):
                try:
                    data_instance = self.pilatesInputDict[(yr, it)]
                    # Access the data using the provided accessor
                    df = self.__accessor(data_instance)
                    if df is not None and not df.empty:
                        data_frames[(yr, it)] = df
                        print(
                            f" - Loaded year {yr}, iteration {it} ({df.shape[0]} rows)"
                        )
                    else:
                        print(f" - No data for year {yr}, iteration {it}")
                except Exception as e:
                    print(f" - Error loading data for year {yr}, iteration {it}: {e}")
                    # Optionally log the error more verbosely

        if not data_frames:
            print(f"No data found across any iteration for {self.__class__.__name__}.")

            # Determine expected index names from the 'columns' attribute
            # The concatenated DF should have index [year, iteration] + self.__columns
            concat_names_if_empty = ["year", "iteration"] + self.__columns

            # Return empty DF with expected index names
            return pd.DataFrame(columns=[]).set_index(
                pd.MultiIndex.from_tuples([], names=concat_names_if_empty)
            )

        # Concatenate DataFrames
        # Use index names including 'year' and 'iteration' plus the accessor's DF index names

        # Get the index names from the first loaded dataframe if available
        first_df_index = next(iter(data_frames.values())).index
        first_df_index_names = list(
            first_df_index.names or []
        )  # Use list() to handle None

        # Filter out None from index names before concatenating
        valid_index_names = [name for name in first_df_index_names if name is not None]

        concat_names = ["year", "iteration"] + valid_index_names

        # Handle cases where the accessor returns a Series (index.names is [None] or just None)
        if not valid_index_names and isinstance(
            next(iter(data_frames.values())), pd.Series
        ):
            # If Series, convert to DataFrame, column name defaults to Series name
            series_name = next(iter(data_frames.values())).name
            if series_name is None:
                series_name = "value"  # Default name if series has no name
            concat_names = ["year", "iteration", series_name]

            # Convert Series to DataFrame for concatenation consistency
            data_frames = {
                k: v.to_frame(name=series_name) for k, v in data_frames.items()
            }  # Convert Series to DF with a column name
        elif not valid_index_names and isinstance(
            next(iter(data_frames.values())), pd.DataFrame
        ):
            # If DF with unnamed index, columns should be preserved.
            # pd.concat handles column alignment automatically.
            pass  # Keep concat_names as ["year", "iteration"] + [] = ["year", "iteration"]

        print(f"Concatenating dataframes with names: {concat_names}")

        combined_df = pd.concat(
            data_frames, names=concat_names, axis=0  # Concatenate rows
        )

        # Set the indexedOn attribute based on the concatenated DataFrame's index names
        # Ensure index has names before converting to list
        self.indexedOn = (
            list(combined_df.index.names) if combined_df.index.names else []
        )

        print(
            f"Finished loading {self.__class__.__name__}. Result shape: {combined_df.shape}"
        )

        return combined_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        # Base class preprocess does nothing by default

        return df


class TripPMTByCountyByYear(TAZBasedDataFrame, InfoByYear):
    """
    Aggregates TripPMTByOrigin by county and year.
    Inherits from TAZBasedDataFrame (for spatial processing) and InfoByYear (for year aggregation).
    MRO: TripPMTByCountyByYear -> TAZBasedDataFrame -> InfoByYear -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "ActivitySimRunOutputData"
        ],  # Needed by InfoByYear
    ):
        # Define the accessor function to get the data for a single year/iteration
        # This accessor performs aggregation by county and trip_mode.
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access tripPMTByOrigin. Its index is ['trip_mode', 'origin'].
            # We want to aggregate this by ['county', 'trip_mode'].
            # tripPMTByOrigin has a load() that merges with skims and preprocess() that groups by ['trip_mode', 'origin'].
            # It does NOT inherently have geometry/county information added.
            # We need to access its dataFrame and then perform the county aggregation here.

            pmt_by_origin_df = outputData.tripPMTByOrigin.dataFrame

            if pmt_by_origin_df is None or pmt_by_origin_df.empty:
                print(
                    f"Warning: TripPMTByOrigin data is empty or None for run {outputData.inputDirectory.directoryPath}."
                )
                # Return empty DF with expected index levels ['county', 'trip_mode'] and column 'distanceInMiles'
                return pd.DataFrame(
                    columns=["distanceInMiles"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Use the geometry from the current ASim output data instance
            geom = outputData.geometry
            if geom is None or geom.gdf is None:
                print(
                    f"Warning: Geometry not available for ASIM run {outputData.inputDirectory.directoryPath}. Cannot aggregate by county."
                )
                # Return empty DF with expected index levels ['county', 'trip_mode'] and column 'distanceInMiles'
                return pd.DataFrame(
                    columns=["distanceInMiles"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Merge with geometry to get county based on origin TAZ
            # The pmt_by_origin_df index is ['trip_mode', 'origin']
            # The merge needs to happen on the 'origin' level of the index.
            # Reset index to turn levels into columns for merging.
            df_merged = pmt_by_origin_df.reset_index().merge(
                geom.gdf[[geom.index, "county"]],  # Select geoIndex and county from gdf
                left_on="origin",  # Merge using the 'origin' column from the reset index
                right_on=geom.index,  # Merge using the geoIndex column from the gdf
                how="left",  # Keep all rows from pmt_by_origin_df
                suffixes=("", "_geom"),  # Add suffix to gdf columns just in case
            )

            # Drop the duplicate merge key column from the right side if it exists and has a suffix
            if f"{geom.index}_geom" in df_merged.columns and "origin" != geom.index:
                df_merged.drop(columns=[f"{geom.index}_geom"], inplace=True)

            # Ensure the 'county' column exists after merge
            if "county" not in df_merged.columns:
                print(
                    f"Error: 'county' column not added after merging with geometry for run {outputData.inputDirectory.directoryPath}. Cannot aggregate by county."
                )
                # Return empty DF with expected index levels ['county', 'trip_mode'] and column 'distanceInMiles'
                return pd.DataFrame(
                    columns=["distanceInMiles"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Aggregate by county and trip mode
            # Ensure distanceInMiles is numeric before summing
            df_merged["distanceInMiles"] = pd.to_numeric(
                df_merged["distanceInMiles"], errors="coerce"
            ).fillna(0)

            aggregated_df = df_merged.groupby(["county", "trip_mode"]).agg(
                {"distanceInMiles": "sum"}
            )

            # Index names will be ['county', 'trip_mode'] after groupby.
            # Ensure index names are set correctly (groupby should handle this).
            aggregated_df.index.set_names(["county", "trip_mode"], inplace=True)

            print(
                f"Finished accessor aggregation for run {outputData.inputDirectory.directoryPath} ({aggregated_df.shape[0]} rows)."
            )
            return aggregated_df

        # The index names of the accessor's output DF are ['county', 'trip_mode'].
        columns_for_info_by = [
            "county",
            "trip_mode",
        ]  # Expected index names from accessor output

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory)
        # and TAZBasedDataFrame-specific keyword-only arguments (geometry, geoIndex)
        # and InfoByYear-specific keyword-only arguments (pilatesInputDict, accessor, columns)
        # to the superclass, following the MRO.
        super().__init__(
            # Args needed by TAZBasedDataFrame.__init__ (outputDataDirectory, inputDirectory, *, geometry, geoIndex, ...)
            outputDataDirectory=outputDataDirectory,
            inputDirectory=pilatesRunInputDirectory,  # Use Pilates dir as base input dir
            geometry=pilatesRunInputDirectory.geometry,  # Use Pilates geometry
            geoIndex=pilatesRunInputDirectory.geometry.index,  # Use Pilates geo index
            # Args needed by InfoByYear.__init__ (outputDataDirectory, inputDirectory, *, pilatesInputDict, accessor, columns, ...)
            # These also need outputDataDirectory and inputDirectory, already provided above.
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
            # Pass original extra args/kwargs
            # *args, # No extra args expected by this class based on signature
            # **kwargs, # No extra kwargs expected by this class based on signature
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # Attributes like __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear
        # Attributes like geometry and geoIndex are handled by TAZBasedDataFrame

        # The final index after load (from InfoByYear) will be [year, county, trip_mode].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "county", "trip_mode"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear. It calls the accessor defined above.
    # Preprocess method is inherited from TAZBasedDataFrame. The accessor already
    # performs the county aggregation, so the TAZBasedDataFrame.process method
    # on the loaded dataframe (indexed by [year, county, trip_mode]) might not be
    # needed or might need to be used carefully depending on what further processing
    # is desired. By default, TAZBasedDataFrame.preprocess does nothing, which is fine.


class MandatoryLocationByTazByYear(TAZBasedDataFrame, InfoByYear):
    """
    Aggregates MandatoryLocationsByTaz by year.
    Inherits from TAZBasedDataFrame (for spatial processing) and InfoByYear (for year aggregation).
    MRO: MandatoryLocationByTazByYear -> TAZBasedDataFrame -> InfoByYear -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "ActivitySimRunOutputData"
        ],  # Needed by InfoByYear
        geometry: Geometry,  # Needed by TAZBasedDataFrame
    ):
        # Define the accessor function to get the data for a single year/iteration
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access the mandatoryLocationsByTaz dataframe.
            # Its load/preprocess method calculates population and jobs and indexes by geoIndex (e.g., 'TAZ').
            # So the accessor returns a DataFrame indexed by the geoIndex.
            return outputData.mandatoryLocationsByTaz.dataFrame

        # The index name of the accessor's output DF is the geoIndex (e.g., 'TAZ').
        columns_for_info_by = [
            geometry.index
        ]  # Expected index names from accessor output

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory)
        # and TAZBasedDataFrame-specific keyword-only arguments (geometry, geoIndex)
        # and InfoByYear-specific keyword-only arguments (pilatesInputDict, accessor, columns)
        # to the superclass, following the MRO.
        super().__init__(
            # Args needed by TAZBasedDataFrame.__init__ (outputDataDirectory, inputDirectory, *, geometry, geoIndex, ...)
            outputDataDirectory=outputDataDirectory,
            inputDirectory=pilatesRunInputDirectory,  # Use Pilates dir as base input dir
            geometry=geometry,  # Keyword-only for TAZBasedDataFrame
            geoIndex=geometry.index,  # Keyword-only for TAZBasedDataFrame
            # Args needed by InfoByYear.__init__ (outputDataDirectory, inputDirectory, *, pilatesInputDict, accessor, columns, ...)
            # These also need outputDataDirectory and inputDirectory, already provided above.
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
            # Pass original extra args/kwargs (none expected for this specific class)
            # *args,
            # **kwargs,
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # Attributes like __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear
        # Attributes like geometry and geoIndex are handled by TAZBasedDataFrame

        # The final index after load (from InfoByYear) will be [year, geoIndex].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", self.geoIndex] # Set by InfoByYear.load()

    # The load method is inherited from InfoByYear. It calls the accessor defined above.
    # The preprocess method is inherited from TAZBasedDataFrame.
    # When called on the loaded dataframe (indexed by [year, geoIndex]),
    # TAZBasedDataFrame.preprocess (which is the default no-op here) is used.
    # The TAZBasedDataFrame.process method can be used *after* loading/preprocessing
    # to perform spatial aggregations/normalizations on the final dataframe.
    # e.g., mandatoryLocationsByTazByYear.process(aggregateBy=['county', 'year'])


class TripModeCountByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        def accessor(outputData: "ActivitySimRunOutputData") -> pd.DataFrame:
            # Access tripModeCount. Its index name is 'trip_mode'.
            # The accessor returns a DataFrame indexed by 'trip_mode'.
            return outputData.tripModeCount.dataFrame

        columns_for_info_by = [
            "trip_mode"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByIteration-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByIteration parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByIteration
            accessor=accessor,  # Keyword-only for InfoByIteration
            columns=columns_for_info_by,  # Keyword-only for InfoByIteration
        )

        # pilatesInputDict is already stored by InfoByIteration.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByIteration) will be [year, iteration, trip_mode].
        # The indexedOn attribute will be set by InfoByIteration's load() method.
        # self.indexedOn = ["year", "iteration", "trip_mode"] # Set by InfoByIteration.load()

    # Load method is inherited from InfoByIteration.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class TourModeCountByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "ActivitySimRunOutputData"],
    ):
        def accessor(outputData: "ActivitySimRunOutputData") -> pd.DataFrame:
            # Access tourModeCount. Its index name is 'tour_mode'.
            # The accessor returns a DataFrame indexed by 'tour_mode'.
            return outputData.tourModeCount.dataFrame

        columns_for_info_by = [
            "tour_mode"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByIteration-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByIteration parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByIteration
            accessor=accessor,  # Keyword-only for InfoByIteration
            columns=columns_for_info_by,  # Keyword-only for InfoByIteration
        )


class ReplanningEventReasonByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access replanningEventReasons. Its dataFrame is the raw event reasons.
            # We want to count occurrences of each 'reason'.
            df = outputData.replanningEventReasons.dataFrame

            if df is None or df.empty:
                # Return empty DF with expected index name 'reason' and column 'count'
                return pd.DataFrame(
                    columns=["count"], index=pd.Index([], name="reason")
                )

            # Count occurrences of each reason
            if "reason" in df.columns:
                return df["reason"].value_counts().to_frame("count")
            else:
                df.columns.set_names("reason", inplace=True)
                df = df.T
                df.columns = ["count"]
                # Return empty DF with expected index name 'reason' and column 'count'
                return df

        columns_for_info_by = [
            "reason"
        ]  # Expected index name of the aggregated dataframe

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByIteration-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByIteration parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByIteration
            accessor=accessor,  # Keyword-only for InfoByIteration
            columns=columns_for_info_by,  # Keyword-only for InfoByIteration
        )


class ScoreStatsByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access scoreStats. Its dataFrame is the raw score stats file.
            # The structure of this file (scorestats.txt) isn't explicitly processed
            # in a dedicated preprocess method in ScoreStats.
            # We assume the raw file is the desired output for the accessor.
            return outputData.scoreStats.dataFrame

        # The index/columns depend on the raw scorestats.txt file format.
        columns_for_info_by = (
            []
        )  # No specific index levels expected from accessor's output DataFrame

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByIteration-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByIteration parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByIteration
            accessor=accessor,  # Keyword-only for InfoByIteration
            columns=columns_for_info_by,  # Keyword-only for InfoByIteration
        )

        # pilatesInputDict is already stored by InfoByIteration.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant
        # self.__yearToDataFrame = dict() # Handled by InfoByIteration

        # The final index after load (from InfoByIteration) will be [year, iteration, original_integer_index].
        # The indexedOn attribute will be set by InfoByIteration's load() method.
        # self.indexedOn = ["year", "iteration", None] # Set by InfoByIteration.load()

    # Load method is inherited from InfoByIteration.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class TripModeCountByYear(InfoByYear):
    """
    Aggregates TripModeCount by year.
    Inherits from InfoByYear.
    MRO: TripModeCountByYear -> InfoByYear -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "ActivitySimRunOutputData"
        ],  # Needed by InfoByYear
    ):
        # Define the accessor function to get the data for a single year/iteration
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access tripModeCount. Its index name is 'trip_mode'.
            # The accessor returns a DataFrame indexed by 'trip_mode'.
            return outputData.tripModeCount.dataFrame

        columns_for_info_by = [
            "trip_mode"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByYear) will be [year, trip_mode].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "trip_mode"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class TourModeCountByYear(InfoByYear):
    """
    Aggregates TourModeCount by year.
    Inherits from InfoByYear.
    MRO: TourModeCountByYear -> InfoByYear -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "ActivitySimRunOutputData"
        ],  # Needed by InfoByYear
    ):
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access tourModeCount. Its index name is 'tour_mode'.
            # The accessor returns a DataFrame indexed by 'tour_mode'.
            return outputData.tourModeCount.dataFrame

        columns_for_info_by = [
            "tour_mode"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByYear) will be [year, tour_mode].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "tour_mode"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class TripModeCountByCountyByYear(TAZBasedDataFrame, InfoByYear):
    """
    Aggregates TripModeCountByOrigin by county and year.
    Inherits from TAZBasedDataFrame (for spatial processing) and InfoByYear (for year aggregation).
    MRO: TripModeCountByCountyByYear -> TAZBasedDataFrame -> InfoByYear -> OutputDataFrame -> object
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, TAZBasedDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "ActivitySimRunOutputData"
        ],  # Needed by InfoByYear
    ):
        # Define the accessor function to get the data for a single year/iteration
        # This accessor performs aggregation by county and trip_mode.
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            # Access tripModeCountByOrigin. Its index is ['trip_mode', 'origin'].
            # We want to aggregate this by ['county', 'trip_mode'].
            # tripModeCountByOrigin is a TripModeCount subclass, which is a TAZBasedDataFrame.
            # It has a process method that can aggregate by county using its geometry.
            # Use its dataFrame property (which is indexed by ['trip_mode', 'origin'])
            # and call its process method to aggregate by county and trip_mode.

            df = outputData.tripModeCountByOrigin.dataFrame

            if df is None or df.empty:
                print(
                    f"Warning: TripModeCountByOrigin data is empty or None for run {outputData.inputDirectory.directoryPath}."
                )
                # Return empty DF with expected index levels ['county', 'trip_mode'] and column 'count'
                return pd.DataFrame(
                    columns=["count"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Call the process method on the TripModeCountByOrigin instance.
            # This instance (outputData.tripModeCountByOrigin) already has the geometry and geoIndex ('origin') set.
            # Its process method knows how to merge with geometry based on its geoIndex.
            # We want to aggregate by 'county' and 'trip_mode'. 'county' comes from geometry, 'trip_mode' is an index level.
            try:
                aggregated_df = outputData.tripModeCountByOrigin.process(
                    normalize=dict(),  # No normalization
                    aggregateBy=[
                        "county",
                        "trip_mode",
                    ],  # Group by county (from geo merge) and trip_mode (index level)
                    mapping={"count": "sum"},  # Sum the counts
                )

                # The result should be indexed by ['county', 'trip_mode']
                # Ensure index names are set explicitly by process or here.
                if not isinstance(aggregated_df.index, pd.MultiIndex) or list(
                    aggregated_df.index.names
                ) != ["county", "trip_mode"]:
                    print(
                        f"Warning: Accessor aggregation did not result in expected index ['county', 'trip_mode'] for run {outputData.inputDirectory.directoryPath}. Actual index names: {list(aggregated_df.index.names)}"
                    )
                    # Attempt to reset/set index
                    if all(
                        col in aggregated_df.columns for col in ["county", "trip_mode"]
                    ):
                        aggregated_df = aggregated_df.reset_index().set_index(
                            ["county", "trip_mode"]
                        )
                        aggregated_df.index.set_names(
                            ["county", "trip_mode"], inplace=True
                        )
                    else:
                        print(
                            "Error: Cannot set index ['county', 'trip_mode'] as required columns are missing after aggregation."
                        )
                        return pd.DataFrame(
                            columns=["count"],
                            index=pd.MultiIndex.from_tuples(
                                [], names=["county", "trip_mode"]
                            ),
                        )

                print(
                    f"Finished accessor aggregation for run {outputData.inputDirectory.directoryPath} ({aggregated_df.shape[0]} rows)."
                )
                return aggregated_df

            except Exception as e:
                print(
                    f"Error during accessor aggregation for run {outputData.inputDirectory.directoryPath}: {e}"
                )
                # Return empty DF on error with expected structure
                return pd.DataFrame(
                    columns=["count"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

        # The index names of the accessor's output DF are ['county', 'trip_mode'].
        columns_for_info_by = [
            "county",
            "trip_mode",
        ]  # Expected index names from accessor output

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory)
        # and TAZBasedDataFrame-specific keyword-only arguments (geometry, geoIndex)
        # and InfoByYear-specific keyword-only arguments (pilatesInputDict, accessor, columns)
        # to the superclass, following the MRO.
        super().__init__(
            # Args needed by TAZBasedDataFrame.__init__ (outputDataDirectory, inputDirectory, *, geometry, geoIndex, ...)
            outputDataDirectory=outputDataDirectory,
            inputDirectory=pilatesRunInputDirectory,  # Use Pilates dir as base input dir
            geometry=pilatesRunInputDirectory.geometry,  # Use Pilates geometry
            # Note: The accessor aggregates *to* county, not *from* the TAZ index directly.
            # The geoIndex for this class as a whole is less clear. It's aggregating *by* county.
            # The TAZBasedDataFrame.process method is called *within* the accessor on the TripModeCountByOrigin DF.
            # The geometry and geoIndex passed here are for *this* class if its own process method is called later.
            # Let's use the main Pilates geometry and its default index.
            geoIndex=pilatesRunInputDirectory.geometry.index,  # Use main Pilates geo index name
            # Args needed by InfoByYear.__init__ (outputDataDirectory, inputDirectory, *, pilatesInputDict, accessor, columns, ...)
            # These also need outputDataDirectory and inputDirectory, already provided above.
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
            # Pass original extra args/kwargs
            # *args, # No extra args expected by this class based on signature
            # **kwargs, # No extra kwargs expected by this class based on signature
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # Attributes like __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear
        # Attributes like geometry and geoIndex are handled by TAZBasedDataFrame

        # The final index after load (from InfoByYear) will be [year, county, trip_mode].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "county", "trip_mode"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear. It calls the accessor defined above.
    # Preprocess method is inherited from TAZBasedDataFrame.
    # When called on the loaded dataframe (indexed by [year, county, trip_mode]),
    # TAZBasedDataFrame.preprocess (which is the default no-op here) is used.
    # The TAZBasedDataFrame.process method can be used *after* loading/preprocessing
    # to perform further spatial aggregations/normalizations if needed.


class ModeVMTByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Needed by OutputDataFrame, InfoByYear
        pilatesRunInputDirectory: PilatesRunInputDirectory,  # Needed by OutputDataFrame, InfoByYear
        pilatesInputDict: Dict[
            Tuple[int, int], "BeamRunOutputData"
        ],  # Needed by InfoByYear
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access the modeVMT dataframe. Its index name is 'mode_extended'.
            # The dataframe has one column, 'vehicleMiles'.
            # The accessor returns a DataFrame indexed by 'mode_extended'.
            return outputData.modeVMT.dataFrame

        columns_for_info_by = [
            "mode_extended"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByYear) will be [year, mode_extended].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "mode_extended"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class TripsByYear(InfoByYear):
    """
    Aggregates aggregated trips data by year.
    Inherits from InfoByYear.
    MRO: TripsByYear -> InfoByYear -> OutputDataFrame -> object

    Attributes:

        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.

        pilatesRunInputDirectory (PilatesRunInputDirectory): The Pilates run input directory.

        pilatesInputDict (Dict[Tuple[int, int], "BeamRunOutputData"]): A dictionary mapping years to corresponding BeamRunOutputData instances.

        indexedOn (str): The column used as the index for the DataFrame.

    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        # Define the accessor function to get the data for a single year/iteration
        # This accessor needs to call the getAggregatedTrips method on the BeamOutputData instance.
        # The result is a DataFrame of trips indexed by default integer index, with a 'tripId' column etc.
        # The structure is the output of mergeWithTripsAndAggregate.
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Need the corresponding ASIM run input directory to pass to getAggregatedTrips
            # This is complex. The PilatesRunInputDirectory object holds all ASIM runs.
            # We can access the specific ASIM input run directory using the same year/iteration key.
            asim_input_dir = pilatesRunInputDirectory.asimRuns.get(
                (outputData.inputDirectory.year, outputData.inputDirectory.iteration)
            )
            if asim_input_dir is None:
                print(
                    f"Warning: Corresponding ASIM input directory not found for Beam run {outputData.inputDirectory.directoryPath}. Cannot aggregate trips."
                )
                return pd.DataFrame()  # Return empty DF

            # Call the getAggregatedTrips method on the BeamOutputData instance
            # Pass the corresponding ASIM input directory
            print(
                f"Calling getAggregatedTrips for {outputData.inputDirectory.directoryPath}..."
            )
            aggregated_trips_df = outputData.getAggregatedTrips(asim_input_dir)
            print(
                f"getAggregatedTrips returned shape {aggregated_trips_df.shape if aggregated_trips_df is not None else 'None'}."
            )
            return aggregated_trips_df

        # The accessor's output DataFrame is the result of mergeWithTripsAndAggregate.
        # This result is indexed by default integer index and has columns like 'tripId', 'person_id', etc.
        # The `columns` list for InfoByYear should describe the index *of the accessor's output*.
        # Since it has a default integer index (index.name is None), we set columns=[] as in ScoreStatsByIteration.
        columns_for_info_by = []

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # __lastIterationPerYear and __yearToDataFrame are handled by InfoByYear

        # The final index after load (from InfoByYear) will be [year, original_integer_index].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", None] # Set by InfoByYear.load()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method is called *after* loading the dataFrame property of the base class (__InfoByYear).
        # The base class loads and concatenates the aggregated trips data frames across years.
        # Any necessary preprocessing on the combined DataFrame would happen here.
        # For now, returning the concatenated data frame as is, assuming aggregation happened in the accessor.

        print(f"Preprocessing TripsByYear DataFrame with shape: {df.shape}")

        # Add any necessary post-processing here, e.g., calculate overall stats, add columns, etc.

        # For example, maybe calculate total trips per year?
        # df['total_trips'] = 1 # Example
        # total_trips_by_year = df.groupby('year').agg(total_trips=('total_trips', 'sum'))
        # But preprocess should return a DataFrame with the same index as the input df.
        # So maybe calculate things and add as new columns? Or just return as is.
        # Let's return as is for now.

        return df  # Return the dataframe


class ModeEnergyByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access the modeEnergy dataframe. Its index name is 'mode_extended'.
            # The accessor returns a DataFrame indexed by 'mode_extended'.
            return outputData.modeEnergy.dataFrame

        columns_for_info_by = [
            "mode_extended"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByYear) will be [year, mode_extended].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "mode_extended"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class ModePMTByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access the modePMT dataframe. Its index name is 'mode_extended'.
            # The accessor returns a DataFrame indexed by 'mode_extended'.
            return outputData.modePMT.dataFrame

        columns_for_info_by = [
            "mode_extended"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByYear-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByYear parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByYear
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByYear
            accessor=accessor,  # Keyword-only for InfoByYear
            columns=columns_for_info_by,  # Keyword-only for InfoByYear
        )

        # pilatesInputDict is already stored by InfoByYear.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByYear) will be [year, mode_extended].
        # The indexedOn attribute will be set by InfoByYear's load() method.
        # self.indexedOn = ["year", "mode_extended"] # Set by InfoByYear.load()

    # Load method is inherited from InfoByYear.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class ModePMTByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access the modePMT dataframe. Its index name is 'mode_extended'.
            # The accessor returns a DataFrame indexed by 'mode_extended'.
            return outputData.modePMT.dataFrame

        columns_for_info_by = [
            "mode_extended"
        ]  # Expected index name of the accessor's output DF

        # Pass common arguments (outputDataDirectory, pilatesRunInputDirectory) and
        # InfoByIteration-specific arguments (pilatesInputDict, accessor, columns)
        # to the InfoByIteration parent.
        super().__init__(
            outputDataDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesRunInputDirectory,  # Positional for OutputDataFrame / InfoByIteration
            pilatesInputDict=pilatesInputDict,  # Keyword-only for InfoByIteration
            accessor=accessor,  # Keyword-only for InfoByIteration
            columns=columns_for_info_by,  # Keyword-only for InfoByIteration
        )

        # pilatesInputDict is already stored by InfoByIteration.__init__
        # self.pilatesInputDict = pilatesInputDict # Redundant

        # The final index after load (from InfoByIteration) will be [year, iteration, mode_extended].
        # The indexedOn attribute will be set by InfoByIteration's load() method.
        # self.indexedOn = ["year", "iteration", "mode_extended"] # Set by InfoByIteration.load()

    # Load method is inherited from InfoByIteration.
    # Preprocess method is inherited from OutputDataFrame (default no-op).


class CongestionInfoByYear(TAZBasedDataFrame, InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            tazIndex = outputData.geometry.index

            df = outputData.tazTrafficVolumes.dataFrame

            if df is None or df.empty:
                # Return empty DF with expected index levels ['roadType', 'TAZ']

                return pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([], names=["roadType", tazIndex])
                )

            df["mph"] = df["VMT"] / df["VHT"]

            # Use the constant for the congestion threshold

            df["congestedHours"] = (df["mph"] < CONGESTION_THRESHOLD_MPH).astype(
                int
            )  # convert boolean to int (0 or 1)

            # df is indexed by [TAZ, hour, attributeOrigType]

            df = df.groupby([tazIndex, "attributeOrigType"]).agg(
                {"VMT": "sum", "VHT": "sum", "congestedHours": "sum"}
            )

            # Recalculate mph after aggregation

            df["mph"] = df["VMT"] / df["VHT"]

            # Unstack attributeOrigType to columns if needed, or keep as index level

            # The original code unstacked TAZ, let's stick to that pattern if intended

            # df = df.unstack(tazIndex) # This would make TAZ columns, metric rows

            # Let's assume the desired output index for this aggregation is [roadType, TAZ]

            df = df.unstack("attributeOrigType")  # Unstack road type to columns

            # The index is now TAZ. Columns are a MultiIndex [metric, roadType].
            # The original code then called columns.set_names("metric", level=0).
            # This seems to imply the desired format was TAZ index, and (metric, roadType) columns.
            # The __InfoByYear/Iteration base class expects the index of the accessor's DF in `columns`.
            # The index here is TAZ. So columns should be [tazIndex].
            # Re-reading the original code: df.unstack(tazIndex) was done.
            # Let's follow the original logic in the accessor:
            df_unstacked = df.unstack(tazIndex)
            if isinstance(df_unstacked, pd.DataFrame):
                df_unstacked.columns.set_names("metric", level=0, inplace=True)
            elif isinstance(df_unstacked, pd.Series):
                df_unstacked = df_unstacked.index.set_names(
                    "metric", level=0
                ).to_frame()
            # The index of this DF is now attributeOrigType.
            # So the expected index names are ['attributeOrigType'].
            return df_unstacked

        # The index name of the accessor's output DF is 'attributeOrigType'.
        columns = ["attributeOrigType"]  # Corrected based on accessor's output index

        InfoByYear.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict,
            accessor,
            columns,
        )

        # Then call TAZBasedDataFrame.__init__

        # Pass geometry from the Pilates run input directory

        TAZBasedDataFrame.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            geometry=pilatesRunInputDirectory.geometry,
            geoIndex=pilatesRunInputDirectory.geometry.index
            # pilatesInputDict=pilatesInputDict,
            # accessor=accessor,
            # columns=columns,
        )

        self.pilatesInputDict = pilatesInputDict


class CongestionInfoByIteration(TAZBasedDataFrame, InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            tazIndex = outputData.geometry.index

            df = outputData.tazTrafficVolumes.dataFrame

            if df is None or df.empty:
                # Return empty DF with expected index levels ['roadType', 'TAZ']

                return pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([], names=["roadType", tazIndex])
                )

            df["mph"] = df["VMT"] / df["VHT"]

            # Use the constant for the congestion threshold

            df["congestedHours"] = (df["mph"] < CONGESTION_THRESHOLD_MPH).astype(
                int
            )  # convert boolean to int (0 or 1)

            # df is indexed by [TAZ, hour, attributeOrigType]

            df = df.groupby([tazIndex, "attributeOrigType"]).agg(
                {"VMT": "sum", "VHT": "sum", "congestedHours": "sum"}
            )
            # Recalculate mph after aggregation
            df["mph"] = df["VMT"] / df["VHT"]
            # Follow original pattern: unstack TAZ
            df_unstacked = df.unstack(tazIndex)
            df_unstacked.columns.set_names("metric", level=0, inplace=True)
            # The index of this DF is now attributeOrigType.
            # So the expected index names are ['attributeOrigType'].
            return df_unstacked

        # The index name of the accessor's output DF is 'attributeOrigType'.
        columns = ["attributeOrigType"]  # Corrected based on accessor's output index
        # Call __InfoByIteration.__init__ first

        InfoByIteration.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict,
            accessor,
            columns,
        )

        # Then call TAZBasedDataFrame.__init__

        # Pass geometry from the Pilates run input directory

        TAZBasedDataFrame.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            geometry=pilatesRunInputDirectory.geometry,
            geoIndex=pilatesRunInputDirectory.geometry.index
            # pilatesInputDict=pilatesInputDict,
            # accessor=accessor,
            # columns=columns,
        )

        self.pilatesInputDict = pilatesInputDict


class PassengerMilesByVehicleAndModeByYear(InfoByYear):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access passengerMilesByVehicleAndMode. Its index is 'vehicleType'.

            # Its columns are the different mode_extended values.

            return outputData.passengerMilesByVehicleAndMode.dataFrame

        # The index name of the accessor's output DF is 'vehicleType'.

        columns = ["vehicleType"]  # Corrected based on accessor's output index

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict,
            accessor,
            columns,
        )


class PassengerMilesByVehicleAndModeByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access passengerMilesByVehicleAndMode. Its index is 'vehicleType'.

            # Its columns are the different mode_extended values.

            return outputData.passengerMilesByVehicleAndMode.dataFrame

        columns = ["vehicleType"]  # Corrected based on accessor's output index

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict,
            accessor,
            columns,
        )


class RealizedModeCountByIteration(InfoByIteration):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            # Access realizedModeCount. Its index name is 'mode'.

            # It has one column, 'RealizedTrips'.

            return outputData.realizedModeCount.dataFrame

        columns = ["mode"]  # Corrected based on accessor's output index

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict,
            accessor,
            columns,
        )
