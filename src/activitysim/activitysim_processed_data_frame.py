from typing import Optional, List

import numpy as np
import pandas as pd

from src.activitysim.activitysim_input_directory import ActivitySimRunOutputDirectory
from src.geometry import Geometry
from src.input_directories import PilatesRunOutputDirectory
from src.processed_data_frame import ProcessedDataFrame
from src.processed_data_frame_mixins import TAZBasedDataFrame
from src.activitysim.activitysim_transformations import filterPersons, filterHouseholds, filterTrips, filterTours


class ProcessedPersonsFile(ProcessedDataFrame):
    """
    Represents a processed persons file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed persons data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunOutputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed persons file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunOutputDirectory,
        *args,  # Accept args/kwargs for MI compatibility
        **kwargs,
    ):
        """
        Initializes a ProcessedPersonsFile instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
            activitySimOutputData (ActivitySimRunOutputDirectory): The ActivitySim output data directory.
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


class ProcessedHouseholdsFile(ProcessedDataFrame):
    """
    Represents a processed households file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed households data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunOutputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed households file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunOutputDirectory,
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


class ProcessedTripsFile(ProcessedDataFrame):
    """
    Represents a processed trips file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed trips data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunOutputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed trips file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunOutputDirectory,
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


class ProcessedToursFile(ProcessedDataFrame):
    """
    Represents a processed tours file derived from ActivitySim output data.

    This class provides functionality to load and preprocess processed tours data obtained from ActivitySim simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        activitySimOutputData (ActivitySimRunOutputDirectory): The ActivitySim output data directory.
        indexedOn (str): The column used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed tours file from the ActivitySim output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        activitySimOutputData: ActivitySimRunOutputDirectory,
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


class ProcessedSkimsFile(ProcessedDataFrame):
    """
    Represents a processed skims file derived from Pilates output data.

    This class provides functionality to load and preprocess processed skims data obtained from Pilates simulations.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesOutputData (PilatesRunOutputDirectory): The Pilates output data directory.
        indexedOn (list): The columns used as the index for the DataFrame.

    Methods:
        preprocess(df): Applies specific preprocessing steps to the input DataFrame.
        load(): Loads the processed skims file from the Pilates output data directory.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesOutputData: PilatesRunOutputDirectory,
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


class TripPMT(ProcessedDataFrame):
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
