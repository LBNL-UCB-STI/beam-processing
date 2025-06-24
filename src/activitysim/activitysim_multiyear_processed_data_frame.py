import hashlib
from typing import Dict, Tuple, Optional, Callable, List
import logging

import pandas as pd

from src.geometry import Geometry
from src.input_directories import PilatesRunOutputDirectory
from src.processed_data_frame import ProcessedDataFrame
from src.processed_data_frame_aggregators import InfoByYear, InfoByIteration
from src.processed_data_frame_mixins import TAZBasedDataFrame
from src.processed_dataframe_agg_base import AggregatedProcessedDataFrameBase

# Set up logging
logger = logging.getLogger(__name__)

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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration
        as required by the base aggregator classes (InfoByYear/InfoByIteration).
        """
        def trip_pmt_by_year_accessor(
            outputData: "ActivitySimRunOutputData", year: int, iteration: int # Added year, iteration
        ) -> Optional[pd.DataFrame]:
            try:
                # Access the dataFrame property of the TripPMT instance
                if hasattr(outputData, 'tripPMT') and outputData.tripPMT is not None:
                    return outputData.tripPMT.dataFrame
                else:
                    logger.warning(f"tripPMT attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing tripPMT data for year {year}: {e}")
                return None
        return trip_pmt_by_year_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on TripPMT definition, the default index is ['trip_mode']
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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def trip_pmt_by_primary_purpose_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Access the dataFrame property of the TripPMTByPrimaryPurpose instance
                if hasattr(outputData, 'tripPMTByPrimaryPurpose') and outputData.tripPMTByPrimaryPurpose is not None:
                    return outputData.tripPMTByPrimaryPurpose.dataFrame
                else:
                     logger.warning(f"tripPMTByPrimaryPurpose attribute not found or is None in {type(outputData).__name__} for year {year}")
                     return None
            except Exception as e:
                logger.error(f"Error accessing tripPMTByPrimaryPurpose data for year {year}: {e}")
                return None
        return trip_pmt_by_primary_purpose_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.

        For TripPMTByPrimaryPurpose, the index is ['trip_mode', 'primary_purpose'].
        """
        return ["trip_mode", "primary_purpose"]

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

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        (TAZ-level PMT data) from a single ModelOutputData instance.
        This function is passed to the aggregator helper.
        The accessor signature must accept outputData, year, and iteration.
        """
        def trip_pmt_by_county_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Access the dataFrame property of the TripPMTByOrigin instance
                if hasattr(outputData, 'tripPMTByOrigin') and outputData.tripPMTByOrigin is not None:
                    pmt_by_origin_df = outputData.tripPMTByOrigin.dataFrame
                else:
                    logger.warning(f"tripPMTByOrigin attribute not found or is None in {type(outputData).__name__} for year {year}")
                    pmt_by_origin_df = None


                if pmt_by_origin_df is None or pmt_by_origin_df.empty:
                    logger.warning(
                        f"TripPMTByOrigin data is empty or None for run {getattr(outputData.inputDirectory, 'directoryPath', 'Unknown')} (Year {year})."
                    )
                    # Return empty DF with expected index levels ['origin', 'trip_mode'] and column 'distanceInMiles'
                    # The accessor should return the data *before* the county aggregation step
                    return pd.DataFrame(
                        columns=["distanceInMiles"],
                        index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                    )

                # Ensure distanceInMiles is numeric before summing
                if "distanceInMiles" not in pmt_by_origin_df.columns:
                     logger.error(f"'distanceInMiles' column not found in TripPMTByOrigin data for run {getattr(outputData.inputDirectory, 'directoryPath', 'Unknown')} (Year {year})")
                     return pd.DataFrame(
                        columns=["distanceInMiles"],
                        index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                    )

                pmt_by_origin_df["distanceInMiles"] = pd.to_numeric(
                    pmt_by_origin_df["distanceInMiles"], errors="coerce"
                ).fillna(0)

                # Group by origin (TAZ) and trip_mode to match _get_expected_index_names
                # The index of TripPMTByOrigin is ['trip_mode', 'origin']
                # Need to reset index to group by columns 'origin' and 'trip_mode'
                if not isinstance(pmt_by_origin_df.index, pd.MultiIndex) or list(pmt_by_origin_df.index.names) != ["trip_mode", "origin"]:
                     logger.warning(f"Unexpected index structure for TripPMTByOrigin: {pmt_by_origin_df.index.names}. Expected ['trip_mode', 'origin']. Attempting to reset index for year {year}.")
                     pmt_by_origin_df = pmt_by_origin_df.reset_index()
                     group_cols = ["origin", "trip_mode"]
                else:
                     # Index is correct, group by index levels
                     group_cols = ["origin", "trip_mode"]


                # Ensure group columns exist after potential reset
                if not all(col in pmt_by_origin_df.columns for col in group_cols):
                     logger.error(f"Required grouping columns {group_cols} not found after index reset for TripPMTByOrigin for year {year}.")
                     return pd.DataFrame(
                        columns=["distanceInMiles"],
                        index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                    )


                aggregated_pmt = pmt_by_origin_df.groupby(group_cols).agg(
                    {"distanceInMiles": "sum"}
                )

                logger.debug(
                    f"Finished accessor processing for run {getattr(outputData.inputDirectory, 'directoryPath', 'Unknown')} (Year {year}) ({aggregated_pmt.shape[0]} rows)."
                )
                return aggregated_pmt
            except Exception as e:
                logger.error(f"Error in trip PMT by county accessor for year {year}: {e}")
                return pd.DataFrame(
                    columns=["distanceInMiles"],
                    index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                )
        return trip_pmt_by_county_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor
        (at the TAZ level before county aggregation).
        """
        # The accessor groups by origin and trip_mode
        return ["origin", "trip_mode"]

    def preprocess(self, df):
        """
        Aggregates the TAZ-level data (df) by county.
        The input df is the data aggregated across years by the base class's load method,
        indexed by the names returned by _get_expected_index_names (i.e., ['origin', 'trip_mode']).
        """
        try:
            taz_agg_df = df

            if taz_agg_df is None or taz_agg_df.empty:
                logger.warning(
                    f"Aggregated TAZ-level data is empty or None for {self.name} during preprocess."
                )
                return pd.DataFrame(
                    index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]),
                    columns=["total_count"] # Assuming the aggregated value column will be named 'total_count'
                )

            # Validate index structure
            expected_index_names = self._get_expected_index_names()
            if not isinstance(taz_agg_df.index, pd.MultiIndex) or list(taz_agg_df.index.names) != expected_index_names:
                logger.warning(f"Unexpected index structure in preprocess for {self.name}. Expected {expected_index_names} but got {taz_agg_df.index.names}. Attempting to reset index.")
                taz_agg_df = taz_agg_df.reset_index()
                # After reset, the columns are the old index names + value columns
                origin_col = 'origin' # Assuming 'origin' is the column name after reset
            else:
                 # Index is correct, 'origin' is the first level
                 origin_col = taz_agg_df.index.names[0] # Should be 'origin'


            # Get geometry from the class instance
            geom = self.geometry

            if geom is None or geom.gdf is None:
                logger.error(f"Geometry not available for {self.name}. Cannot aggregate by county in preprocess.")
                return pd.DataFrame(
                    columns=["total_count"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Ensure the origin column exists after potential reset
            if origin_col not in taz_agg_df.columns:
                 logger.error(f"Origin column '{origin_col}' not found in DataFrame during preprocess for {self.name}.")
                 return pd.DataFrame(
                    columns=["total_count"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Merge with geometry to get county
            # Need to ensure the geometry index column name matches the origin column name in the dataframe
            if origin_col != geom.index:
                 logger.warning(f"Origin column name '{origin_col}' does not match geometry index name '{geom.index}'. Renaming origin column for merge.")
                 df_to_merge = taz_agg_df.rename(columns={origin_col: geom.index})
                 left_on_col = geom.index
            else:
                 df_to_merge = taz_agg_df
                 left_on_col = origin_col


            df_merged = df_to_merge.merge(
                geom.gdf[[geom.index, "county"]],
                left_on=left_on_col,
                right_on=geom.index,
                how="left",
                suffixes=("", "_geom"),
            )

            # Drop duplicate merge key column if it exists and is not the original origin column
            if f"{geom.index}_geom" in df_merged.columns and left_on_col != geom.index:
                 df_merged.drop(columns=[f"{geom.index}_geom"], inplace=True)

            # Check if merge was successful
            if 'county' not in df_merged.columns:
                raise ValueError(f"Merge with geometry failed. 'county' column not found after merge for {self.name}.")

            # Determine value column name
            # The base class aggregator puts the aggregated value in a column named 'value' by default
            value_column = 'value'
            if value_column not in df_merged.columns:
                # Fallback: find the first non-index/non-merge column
                potential_value_columns = [col for col in df_merged.columns if col not in expected_index_names + [origin_col, 'county', geom.index]]
                if potential_value_columns:
                    value_column = potential_value_columns[0]
                    logger.warning(f"Using '{value_column}' column for aggregation as 'value' was not found in {self.name}.")
                else:
                    logger.error(f"No value columns found for aggregation in {self.name}")
                    return pd.DataFrame(
                        index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]),
                        columns=["total_count"]
                    )

            # Ensure value column is numeric
            df_merged[value_column] = pd.to_numeric(df_merged[value_column], errors="coerce").fillna(0)

            # Aggregate by county and trip_mode
            # Need to ensure 'trip_mode' column exists after potential index reset
            if 'trip_mode' not in df_merged.columns:
                 logger.error(f"'trip_mode' column not found in DataFrame during preprocess for {self.name}.")
                 return pd.DataFrame(
                    index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]),
                    columns=["total_count"]
                )

            county_agg_df = df_merged.groupby(['county', 'trip_mode'])[value_column].sum().reset_index()
            county_agg_df = county_agg_df.rename(columns={value_column: 'total_count'})
            county_agg_df = county_agg_df.set_index(['county', 'trip_mode'])

            logger.info(f"Finished county aggregation in preprocess for {self.name} ({county_agg_df.shape[0]} rows).")
            return county_agg_df

        except Exception as e:
            logger.error(f"Error in preprocess for {self.name}: {e}")
            return pd.DataFrame(
                index=pd.MultiIndex.from_product([[], []], names=["county", "trip_mode"]),
                columns=["total_count"]
            )


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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def mandatory_location_by_taz_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Access the dataFrame property of the MandatoryLocationsByTaz instance
                if hasattr(outputData, 'mandatoryLocationsByTaz') and outputData.mandatoryLocationsByTaz is not None:
                    return outputData.mandatoryLocationsByTaz.dataFrame
                else:
                    logger.warning(f"mandatoryLocationsByTaz attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing mandatoryLocationsByTaz data for year {year}: {e}")
                return None
        return mandatory_location_by_taz_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on MandatoryLocationsByTaz definition, the index is the geoIndex (e.g., 'TAZ')
        # Need to access the geometry from the class instance itself, not the accessor's input
        # This requires the accessor to be bound to the class instance, which it is.
        # However, _get_expected_index_names is called during base class __init__
        # before the geometry is fully set up in the subclass.
        # A safer approach is to rely on the known structure or pass geoIndex explicitly.
        # Assuming the geoIndex is always the first index name for TAZBasedDataFrames used here.
        # Or, better, make geoIndex a required init parameter for TAZBasedDataFrame.
        # For now, let's assume the geoIndex is the expected index name.
        # The MandatoryLocationsByTaz preprocess sets the index name to self.geoIndex.
        # The accessor returns this preprocessed data.
        # So the expected index name is the geoIndex.
        # This method is called by the base class AggregatedProcessedDataFrameBase
        # to check the index of the DataFrame returned by the accessor.
        # The accessor returns the dataFrame from MandatoryLocationsByTaz,
        # which is indexed by self.geoIndex.
        # We need access to self.geoIndex here.
        # Since this method is part of the class, self is available.
        if hasattr(self, 'geoIndex') and self.geoIndex:
             return [self.geoIndex]
        else:
             # Fallback if geoIndex isn't set yet or is None
             logger.warning("geoIndex not available when calling _get_expected_index_names for MandatoryLocationByTazByYear. Defaulting to ['TAZ'].")
             return ["TAZ"]


class TripModeCountByIteration(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripModeCount by iteration using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByIteration

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def trip_mode_count_by_iteration_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TripModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                if hasattr(outputData, 'tripModeCount') and outputData.tripModeCount is not None:
                    return outputData.tripModeCount.dataFrame
                else:
                    logger.warning(f"tripModeCount attribute not found or is None in {type(outputData).__name__} for year {year}, iteration {iteration}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing tripModeCount data for iteration {iteration}: {e}")
                return None
        return trip_mode_count_by_iteration_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on TripModeCount definition, the index is ['trip_mode']
        return ["trip_mode"]


class TourModeCountByIteration(AggregatedProcessedDataFrameBase):
    """
    Aggregates TourModeCount by iteration using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByIteration

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def tour_mode_count_by_iteration_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TourModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                if hasattr(outputData, 'tourModeCount') and outputData.tourModeCount is not None:
                    return outputData.tourModeCount.dataFrame
                else:
                    logger.warning(f"tourModeCount attribute not found or is None in {type(outputData).__name__} for year {year}, iteration {iteration}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing tourModeCount data for iteration {iteration}: {e}")
                return None
        return tour_mode_count_by_iteration_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on TourModeCount definition, the index is ['tour_mode']
        return ["tour_mode"]


class TripModeCountByYear(AggregatedProcessedDataFrameBase):
    """
    Aggregates TripModeCount by year using AggregatedProcessedDataFrameBase.
    """

    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        return InfoByYear

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def trip_mode_count_by_year_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Note: TripModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                if hasattr(outputData, 'tripModeCount') and outputData.tripModeCount is not None:
                    return outputData.tripModeCount.dataFrame
                else:
                    logger.warning(f"tripModeCount attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing tripModeCount data for year {year}: {e}")
                return None
        return trip_mode_count_by_year_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on TripModeCount definition, the index is ['trip_mode']
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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def tour_mode_count_by_year_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Note: TourModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                if hasattr(outputData, 'tourModeCount') and outputData.tourModeCount is not None:
                    return outputData.tourModeCount.dataFrame
                else:
                    logger.warning(f"tourModeCount attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing tourModeCount data for year {year}: {e}")
                return None
        return tour_mode_count_by_year_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Based on TourModeCount definition, the index is ['tour_mode']
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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def trips_by_year_accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                # Access the trip_pmt dataFrame
                if hasattr(outputData, 'trip_pmt') and outputData.trip_pmt is not None:
                    trips_df = outputData.trip_pmt.dataFrame
                else:
                    logger.warning(f"trip_pmt attribute not found or is None in {type(outputData).__name__} for year {year}")
                    trips_df = None


                if trips_df is None or trips_df.empty:
                    logger.warning(
                        f"TripPMT data is empty or None for run {getattr(outputData.inputDirectory, 'directoryPath', 'Unknown')} (Year {year})."
                    )
                    # Return an empty DataFrame with expected columns if needed by the aggregator
                    # Assuming trip_pmt has 'distanceInMiles' and is indexed by 'trip_id'
                    return pd.DataFrame(columns=['distanceInMiles'], index=pd.Index([], name='trip_id'))

                # Assuming trip_pmt has relevant columns, e.g., 'distanceInMiles'
                # Select relevant columns and return
                # Need to confirm actual columns available in BeamRunOutputData.trip_pmt
                # For now, returning the whole dataframe as a placeholder
                if 'distanceInMiles' in trips_df.columns:
                    return trips_df[['distanceInMiles']] # Example: select a relevant column
                else:
                    logger.warning(f"'distanceInMiles' column not found in BeamRunOutputData.trip_pmt for run {getattr(outputData.inputDirectory, 'directoryPath', 'Unknown')} (Year {year})")
                    # Return empty DF with expected column and index
                    return pd.DataFrame(columns=['distanceInMiles'], index=pd.Index([], name='trip_id'))

            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.trip_pmt data for year {year}: {e}")
                return None
        return trips_by_year_accessor


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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM trips DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def beam_trips_by_year_accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                 # Access the dataFrame property of the trips instance
                 if hasattr(outputData, 'trips') and outputData.trips is not None:
                    return outputData.trips.dataFrame # Access BEAM trips dataFrame
                 else:
                    logger.warning(f"trips attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.trips data for year {year}: {e}")
                return None
        return beam_trips_by_year_accessor

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

    def _get_accessor(self) -> Callable[["ModelOutputData", int, int], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM skims DataFrame
        from a single ModelOutputData instance.
        The accessor signature must accept outputData, year, and iteration.
        """
        def beam_skims_by_year_accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]: # Added year, iteration
            try:
                 # Access the dataFrame property of the skims instance
                 if hasattr(outputData, 'skims') and outputData.skims is not None:
                    return outputData.skims.dataFrame # Access BEAM skims dataFrame
                 else:
                    logger.warning(f"skims attribute not found or is None in {type(outputData).__name__} for year {year}")
                    return None
            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.skims data for year {year}: {e}")
                return None
        return beam_skims_by_year_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Assuming BEAM skims data is indexed by origin, destination, mode, etc.
        # This needs to be confirmed from the actual data structure.
        return ["origin", "destination", "mode", "time_period"] # Placeholder, needs verification
