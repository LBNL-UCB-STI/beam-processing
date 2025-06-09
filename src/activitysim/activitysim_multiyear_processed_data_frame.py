import hashlib
from typing import Dict, Tuple, Optional, Callable, List

import pandas as pd

from src.geometry import Geometry
from src.input_directories import PilatesRunOutputDirectory
from src.processed_data_frame import ProcessedDataFrame
from src.processed_data_frame_aggregators import InfoByYear, InfoByIteration
from src.processed_data_frame_mixins import TAZBasedDataFrame
from src.processed_dataframe_agg_base import AggregatedProcessedDataFrameBase

# Forward declarations for type hints
OutputDataDirectory = "OutputDataDirectory"
ModelOutputData = "ModelOutputData"
ActivitySimRunOutputData = "ActivitySimRunOutputData"
BeamRunOutputData = "BeamRunOutputData"


class TripPMTByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripPMT by year using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_pmt_accessor(
            outputData: "ActivitySimRunOutputData",
        ) -> Optional[pd.DataFrame]:
            return outputData.tripPMT.dataFrame
        return trip_pmt_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["trip_mode"]

    def preprocess(self, df):
        # Default no-op preprocess inherited from ProcessedDataFrame is fine,
        # unless specific post-aggregation processing is needed.
        return df


class TripPMTByPrimaryPurposeByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripPMTByPrimaryPurpose by year using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_pmt_by_purpose_accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
             return outputData.tripPMTByPrimaryPurpose.dataFrame
        return trip_pmt_by_purpose_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        
        For TripPMTByCountyByYear, the accessor returns data aggregated by origin (TAZ)
        and trip mode *before* the final county aggregation.
        """
        return ["origin", "trip_mode"]

    def preprocess(self, df):
        # Default no-op preprocess inherited from ProcessedDataFrame is fine,
        # unless specific post-aggregation processing is needed.
        return df

# TripPMTByCountyByYear - Refactored to use AggregatedProcessedDataFrameBase and TAZBasedDataFrame
class TripPMTByCountyByYear(TAZBasedDataFrame, AggregatedProcessedDataFrameBase):
    """
    Aggregates TripPMTByOrigin by county and year.
    Inherits from TAZBasedDataFrame (for spatial processing) and AggregatedProcessedDataFrameBase
    (for multi-run aggregation and caching).
    """

    # __init__ is inherited from AggregatedProcessedDataFrameBase

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        (TAZ-level PMT data) from a single ModelOutputData instance.
        This function is passed to the aggregator helper.
        """
        def accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            pmt_by_origin_df = outputData.tripPMTByOrigin.dataFrame

            if pmt_by_origin_df is None or pmt_by_origin_df.empty:
                print(
                    f"Warning: TripPMTByOrigin data is empty or None for run {outputData.inputDirectory.directoryPath}."
                )
                # Return empty DF with expected index levels ['origin', 'trip_mode'] and column 'distanceInMiles'
                # The index names here must match _get_expected_index_names
                return pd.DataFrame(
                    columns=["distanceInMiles"],
                    index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                )

            # Ensure distanceInMiles is numeric before summing
            pmt_by_origin_df["distanceInMiles"] = pd.to_numeric(
                pmt_by_origin_df["distanceInMiles"], errors="coerce"
            ).fillna(0)

            # Group by origin (TAZ) and trip_mode to match _get_expected_index_names
            # Note: TripPMTByOrigin is already aggregated by origin and trip_mode,
            # but doing this explicitly ensures the correct structure for the aggregator
            aggregated_pmt = pmt_by_origin_df.groupby(["origin", "trip_mode"]).agg(
                {"distanceInMiles": "sum"}
            )

            print(
                f"Finished accessor processing for run {outputData.inputDirectory.directoryPath} ({aggregated_pmt.shape[0]} rows)."
            )
            return aggregated_pmt # Return TAZ-level aggregated data for this run
        return accessor


    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor
        (at the TAZ level before county aggregation).
        """
        return ["origin", "trip_mode"]

    def preprocess(self, df):
        """
        Aggregates the TAZ-level data (df) by county.
        The input df is the data aggregated across years by the base class's load method,
        indexed by the names returned by _get_expected_index_names (i.e., ['origin', 'trip_mode']).
        """
        taz_agg_df = df # The input df is the TAZ-level aggregated data

        if taz_agg_df is None or taz_agg_df.empty:
             print(
                f"Warning: Aggregated TAZ-level data is empty or None for {self.name} during preprocess."
            )
             # Return an empty DataFrame with the correct index and column names
             return pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]), columns=["total_count"])


        # Merge the TAZ-level aggregated data with county geometry
        # Assuming self.geometry is a DataFrame with 'county' column and TAZ index
        # The TAZ index name in geometry is expected to match the 'origin' index name in taz_agg_df
        # Ensure the index of taz_agg_df is named 'origin' as expected by the merge
        if taz_agg_df.index.name != 'origin':
            # This check might be overly strict if the index is a MultiIndex.
            # Let's check if the first level name is 'origin'.
            if not isinstance(taz_agg_df.index, pd.MultiIndex) or taz_agg_df.index.names[0] != 'origin':
                 print(f"Warning: First index name of input DataFrame to preprocess is '{taz_agg_df.index.names[0] if isinstance(taz_agg_df.index, pd.MultiIndex) else taz_agg_df.index.name}', expected 'origin'. Attempting to proceed.")


        # Use left_index=True for taz_agg_df (indexed by 'origin') and right_on=geom.index for geometry
        geom = self.geometry # Get geometry from the class instance

        if geom is None or geom.gdf is None:
            print(
                f"Warning: Geometry not available for {self.name}. Cannot aggregate by county in preprocess."
            )
            # Return empty DF with expected index levels ['county', 'trip_mode'] and column 'total_count'
            return pd.DataFrame(
                columns=["total_count"],
                index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
            )

        # Merge with geometry to get county
        # The index of taz_agg_df is ['origin', 'trip_mode']
        # Reset index to merge on 'origin' column
        df_merged = taz_agg_df.reset_index().merge(
            geom.gdf[[geom.index, "county"]],  # Select geoIndex and county from gdf
            left_on="origin",  # Merge using the 'origin' column from the reset index
            right_on=geom.index,  # Merge using the geoIndex column from the gdf
            how="left",  # Keep all rows from taz_agg_df
            suffixes=("", "_geom"),  # Add suffix to gdf columns just in case
        )

        # Drop the duplicate merge key column from the right side if it exists and has a suffix
        if f"{geom.index}_geom" in df_merged.columns and "origin" != geom.index:
            df_merged.drop(columns=[f"{geom.index}_geom"], inplace=True)


        # Check if the merge was successful and 'county' column exists
        if 'county' not in df_merged.columns:
             raise ValueError(f"Merge with geometry failed. 'county' column not found after merge for {self.name} during preprocess.")

        # The aggregated data from InfoByYear will have a column named 'value' by default
        # if the accessor returns a Series or a DataFrame with a single unnamed column.
        # After aggregation by InfoByYear, the value column should be named 'value'
        value_column = 'value' # Column name after aggregation by InfoByYear

        if value_column not in df_merged.columns:
             # Fallback if the column name is not 'value'
             # Assuming the accessor returns a single column and InfoByYear names it 'value'
             # If not, we need to inspect the actual output of the accessor/aggregator
             # For now, let's assume 'value' or the first column if 'value' is not found
             if len(df_merged.columns) > 0: # Check if there are any columns
                 # Find the first column that is not an index name ('origin', 'trip_mode', 'county')
                 potential_value_columns = [col for col in df_merged.columns if col not in ['origin', 'trip_mode', 'county']]
                 if potential_value_columns:
                     value_column = potential_value_columns[0]
                     print(f"Warning: '{value_column}' column used for aggregation as 'value' was not found in {self.name} during preprocess.")
                 else:
                      # If no value columns exist, return empty DataFrame
                      return pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]), columns=["total_count"])
             else:
                  # If no columns exist at all, return empty DataFrame
                  return pd.DataFrame(index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]), columns=["total_count"])


        # Ensure the value column is numeric before summing
        df_merged[value_column] = pd.to_numeric(
            df_merged[value_column], errors="coerce"
        ).fillna(0)


        # Aggregate by county and trip_mode, summing the value column
        # The columns in df_merged are ['origin', 'trip_mode', 'county', value_column]
        county_agg_df = df_merged.reset_index().groupby(['county', 'trip_mode'])[value_column].sum().reset_index()


        # Rename the aggregated value column to a standard name like 'total_count'
        county_agg_df = county_agg_df.rename(columns={value_column: 'total_count'})

        # Set the final index
        county_agg_df = county_agg_df.set_index(['county', 'trip_mode'])

        print(
            f"Finished county aggregation in preprocess for {self.name} ({county_agg_df.shape[0]} rows)."
        )

        return county_agg_df # Return the final dataframe


class MandatoryLocationByTazByYear(AggregatedProcessedDataFrameBase, TAZBasedDataFrame):
    """
    Aggregates MandatoryLocationsByTaz by year.
    Inherits from AggregatedProcessedDataFrameBase (for aggregation logic) and TAZBasedDataFrame (for spatial processing).
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def mandatory_location_accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
             return outputData.mandatoryLocationsByTaz.dataFrame
        return mandatory_location_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # The index name of the accessor's output DF is the geoIndex (e.g., 'TAZ').
        return [self.geometry.index]

    # The load method is inherited from AggregatedProcessedDataFrameBase
    # The __init__ method is inherited from AggregatedProcessedDataFrameBase


class TripModeCountByIteration(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripModeCount by iteration using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByIteration

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_mode_count_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
             return outputData.tripModeCount.dataFrame
        return trip_mode_count_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["trip_mode"]

    # The load method is inherited from AggregatedProcessedDataFrameBase
    # The __init__ method is inherited from AggregatedProcessedDataFrameBase

class TourModeCountByIteration(AggregatedProcessedDataFrameBase):
    """
    Aggregates TourModeCount by iteration using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByIteration

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def tour_mode_count_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
             return outputData.tourModeCount.dataFrame
        return tour_mode_count_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["tour_mode"]

    # The load method is inherited from AggregatedProcessedDataFrameBase
    # The __init__ method is inherited from AggregatedProcessedDataFrameBase


class TripModeCountByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripModeCount by year using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_mode_count_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
             return outputData.tripModeCount.dataFrame
        return trip_mode_count_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["trip_mode"]

class TourModeCountByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates TourModeCount by year using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def tour_mode_count_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
             return outputData.tourModeCount.dataFrame
        return tour_mode_count_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["tour_mode"]


class TripsByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates aggregated trips data by year.
    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory where the file is stored.
        pilatesRunInputDirectory (PilatesRunOutputDirectory): The Pilates run input directory.
        pilatesInputDict (Dict[Tuple[int, int], "BeamRunOutputData"]): A dictionary mapping years to corresponding BeamRunOutputData instances.
        indexedOn (str): The column used as the index for the DataFrame.
    """


    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
         """
         Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
         """
         return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_pmt_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
             # Access the trip_pmt dataFrame
            trips_df = outputData.trip_pmt.dataFrame

            if trips_df is None or trips_df.empty:
                print(
                    f"Warning: TripPMT data is empty or None for run {outputData.inputDirectory.directoryPath}."
                )
                # Return an empty DataFrame with expected columns if needed by the aggregator
                return pd.DataFrame() # Or specify columns if known

            # Assuming trip_pmt has relevant columns, e.g., 'distanceInMiles'
            # Select relevant columns and return
            # Need to confirm actual columns available in BeamRunOutputData.trip_pmt
            # For now, returning the whole dataframe as a placeholder
            return trips_df
        return trip_pmt_accessor


    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Need to confirm the index of BeamRunOutputData.trip_pmt
        # Assuming it's indexed by trip_id
        return ["trip_id"] # Placeholder, needs verification


class BeamTripsByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates BEAM trips data by year.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM trips DataFrame
        from a single ModelOutputData instance.
        """
        def beam_trips_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
             return outputData.trips.dataFrame # Access BEAM trips dataFrame
        return beam_trips_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Assuming BEAM trips data is indexed by trip_id
        return ["trip_id"] # Placeholder, needs verification


class BeamSkimsByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates BEAM skims data by year.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM skims DataFrame
        from a single ModelOutputData instance.
        """
        def beam_skims_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
             return outputData.skims.dataFrame # Access BEAM skims dataFrame
        return beam_skims_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Assuming BEAM skims data is indexed by origin, destination, mode, etc.
        # This needs to be confirmed from the actual data structure.
        return ["origin", "destination", "mode", "time_period"] # Placeholder, needs verification
