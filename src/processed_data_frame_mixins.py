from typing import Optional, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

from src.geometry import Geometry
from src.input_base import OutputDirectory
from src.processed_data_frame import ProcessedDataFrame


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
        inputDirectory: OutputDirectory,  # Expected by OutputDataFrame
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
