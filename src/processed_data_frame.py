import gc
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
import geopandas as gpd

import pandas as pd

from src.constants import TMP_DIR
from src.input_base import InputDirectory
from src.geometry import Geometry


# --- Constants ---


class ProcessedDataFrame(ABC):
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

    @abstractmethod
    def load(self):
        """
        Abstract method for loading data into a DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

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

    def process(
        self,
        normalize: Optional[str],
        aggregateBy: Optional[List[str]],
        mapping: Optional[Dict[str, str]],
    ) -> pd.DataFrame:
        raise NotImplementedError("This class does not have process defined")


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


class TAZBasedDataFrame(ProcessedDataFrame):
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

        super().__init__(outputDataDirectory, inputDirectory, *args, **kwargs)


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
            if mapping:
                output_cols = list(mapping.keys())
                if normalize:
                    for col, func in normalize.items():
                        if func == "area":
                            output_cols.append(col + "Density")
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
            temp_reset = temp.reset_index()

            geom_cols = [self.geometry.index, "county", "areatype10"]
            if "gacres" in additionalColumns:
                geom_cols.append("gacres")
            geom_cols_present = [
                col for col in geom_cols if col in self.geometry.gdf.columns
            ]
            # Ensure geoIndex is always included from gdf for merging
            if self.geometry.index not in geom_cols_present:
                geom_cols_present.append(self.geometry.index)

            if self.geoIndex not in temp_reset.columns:
                print(
                    f"Warning: Column '{self.geoIndex}' not found directly after reset_index(). Assuming it was the original index name."
                )
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
                print(
                    "Warning: Mapping provided without aggregateBy. Applying aggregation to the whole DataFrame."
                )
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

        df_to_merge = df.copy()  # Work on a copy
        if isinstance(df_to_merge.index, pd.MultiIndex):
            if self.geoIndex not in df_to_merge.index.names:
                print(
                    f"Error: Geo-index level '{self.geoIndex}' not found in MultiIndex."
                )
                return gpd.GeoDataFrame()  # Cannot proceed

            df_to_merge = df_to_merge.reset_index()
            merge_col_left = self.geoIndex
        elif df_to_merge.index.name == self.geoIndex:
            df_to_merge = df_to_merge.reset_index()
            merge_col_left = self.geoIndex
        else:
            print(
                f"Error: DataFrame index name is not '{self.geoIndex}' and it's not a MultiIndex with that level."
            )
            print(
                "Attempting to merge on the first column after reset_index() if index was unnamed."
            )
            df_to_merge = df_to_merge.reset_index()
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


class InfoByYear(ProcessedDataFrame):
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
        last_iteration_per_year = {}
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
            concat_names_if_empty = ["year"] + self.__columns

            # Return empty DF with expected index names
            return pd.DataFrame(columns=[]).set_index(
                pd.MultiIndex.from_tuples([], names=concat_names_if_empty)
            )

        first_df_index = next(iter(data_frames.values())).index
        first_df_index_names = list(
            first_df_index.names or []
        )  # Use list() to handle None

        # Filter out None from index names before concatenating
        valid_index_names = [name for name in first_df_index_names if name is not None]

        concat_names = ["year"] + valid_index_names

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
            first_df = next(iter(data_frames.values()))
            if first_df.index.name is None and isinstance(
                first_df.index, pd.RangeIndex
            ):
                print(
                    f"Warning: Accessor returned DataFrame with default integer index for {self.__class__.__name__}. Concatenated columns will include original DF columns."
                )
                pass

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


class InfoByIteration(ProcessedDataFrame):
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

        super().__init__(
            outputDataDirectory,  # Positional argument for OutputDataFrame
            inputDirectory,  # Positional argument for OutputDataFrame
            hash_key=override_hash,  # Keyword-only argument for OutputDataFrame
            *args,
            **kwargs,
        )

        self.pilatesInputDict = pilatesInputDict
        self.__accessor = accessor
        self.__columns = columns  # Store expected index names

    def load(self):
        """
        Loads data from all available iterations for each year using the accessor.
        Concatenates the results.
        """
        print(f"Loading data for {self.__class__.__name__} across all iterations...")
        data_frames = {}

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

            concat_names_if_empty = ["year", "iteration"] + self.__columns

            # Return empty DF with expected index names
            return pd.DataFrame(columns=[]).set_index(
                pd.MultiIndex.from_tuples([], names=concat_names_if_empty)
            )

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
            pass

        print(f"Concatenating dataframes with names: {concat_names}")
        combined_df = pd.concat(
            data_frames, names=concat_names, axis=0  # Concatenate rows
        )
        self.indexedOn = (
            list(combined_df.index.names) if combined_df.index.names else []
        )
        print(
            f"Finished loading {self.__class__.__name__}. Result shape: {combined_df.shape}"
        )

        return combined_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

