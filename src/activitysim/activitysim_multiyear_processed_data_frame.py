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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_pmt_by_year_accessor(
            outputData: "ActivitySimRunOutputData",
        ) -> Optional[pd.DataFrame]:
            try:
                return outputData.tripPMT.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tripPMT data: {e}")
                return None
        return trip_pmt_by_year_accessor

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
        def trip_pmt_by_primary_purpose_accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            try:
                return outputData.tripPMTByPrimaryPurpose.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tripPMTByPrimaryPurpose data: {e}")
                return None
        return trip_pmt_by_primary_purpose_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        
        For TripPMTByPrimaryPurposeByYear, the accessor returns data aggregated by origin (TAZ)
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
        def trip_pmt_by_county_accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            try:
                pmt_by_origin_df = outputData.tripPMTByOrigin.dataFrame

                if pmt_by_origin_df is None or pmt_by_origin_df.empty:
                    logger.warning(
                        f"TripPMTByOrigin data is empty or None for run {outputData.inputDirectory.directoryPath}."
                    )
                    # Return empty DF with expected index levels ['origin', 'trip_mode'] and column 'distanceInMiles'
                    return pd.DataFrame(
                        columns=["distanceInMiles"],
                        index=pd.MultiIndex.from_tuples([], names=["origin", "trip_mode"]),
                    )

                # Ensure distanceInMiles is numeric before summing
                pmt_by_origin_df["distanceInMiles"] = pd.to_numeric(
                    pmt_by_origin_df["distanceInMiles"], errors="coerce"
                ).fillna(0)

                # Group by origin (TAZ) and trip_mode to match _get_expected_index_names
                aggregated_pmt = pmt_by_origin_df.groupby(["origin", "trip_mode"]).agg(
                    {"distanceInMiles": "sum"}
                )

                logger.debug(
                    f"Finished accessor processing for run {outputData.inputDirectory.directoryPath} ({aggregated_pmt.shape[0]} rows)."
                )
                return aggregated_pmt
            except Exception as e:
                logger.error(f"Error in trip PMT by county accessor: {e}")
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
                    columns=["total_count"]
                )

            # Validate index structure
            if not isinstance(taz_agg_df.index, pd.MultiIndex) or taz_agg_df.index.names[0] != 'origin':
                logger.warning(f"Unexpected index structure in preprocess for {self.name}. Expected 'origin' as first index level.")

            # Get geometry from the class instance
            geom = self.geometry

            if geom is None or geom.gdf is None:
                logger.error(f"Geometry not available for {self.name}. Cannot aggregate by county in preprocess.")
                return pd.DataFrame(
                    columns=["total_count"],
                    index=pd.MultiIndex.from_tuples([], names=["county", "trip_mode"]),
                )

            # Merge with geometry to get county
            df_merged = taz_agg_df.reset_index().merge(
                geom.gdf[[geom.index, "county"]],
                left_on="origin",
                right_on=geom.index,
                how="left",
                suffixes=("", "_geom"),
            )

            # Drop duplicate merge key column if it exists
            if f"{geom.index}_geom" in df_merged.columns and "origin" != geom.index:
                df_merged.drop(columns=[f"{geom.index}_geom"], inplace=True)

            # Check if merge was successful
            if 'county' not in df_merged.columns:
                raise ValueError(f"Merge with geometry failed. 'county' column not found after merge for {self.name}.")

            # Determine value column name
            value_column = 'value'
            if value_column not in df_merged.columns:
                potential_value_columns = [col for col in df_merged.columns if col not in ['origin', 'trip_mode', 'county']]
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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def mandatory_location_by_taz_accessor(outputData: "ActivitySimRunOutputData") -> Optional[pd.DataFrame]:
            try:
                return outputData.mandatoryLocationsByTaz.dataFrame
            except Exception as e:
                logger.error(f"Error accessing mandatoryLocationsByTaz data: {e}")
                return None
        return mandatory_location_by_taz_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return [self.geometry.index]


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
        def trip_mode_count_by_iteration_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TripModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                return outputData.tripModeCount.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tripModeCount data for iteration: {e}")
                return None
        return trip_mode_count_by_iteration_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def tour_mode_count_by_iteration_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TourModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                return outputData.tourModeCount.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tourModeCount data for iteration: {e}")
                return None
        return tour_mode_count_by_iteration_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance.
        """
        def trip_mode_count_by_year_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TripModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                return outputData.tripModeCount.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tripModeCount data for year: {e}")
                return None
        return trip_mode_count_by_year_accessor

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
        def tour_mode_count_by_year_accessor(outputData: "ActivitySimRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            try:
                # Note: TourModeCount is not iteration-specific in its load method
                # The accessor signature includes year and iteration for consistency with the aggregator base
                return outputData.tourModeCount.dataFrame
            except Exception as e:
                logger.error(f"Error accessing tourModeCount data for year: {e}")
                return None
        return tour_mode_count_by_year_accessor

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
        def trips_by_year_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            try:
                # Access the trip_pmt dataFrame
                trips_df = outputData.trip_pmt.dataFrame

                if trips_df is None or trips_df.empty:
                    logger.warning(
                        f"TripPMT data is empty or None for run {outputData.inputDirectory.directoryPath}."
                    )
                    # Return an empty DataFrame with expected columns if needed by the aggregator
                    # Assuming trip_pmt has 'distanceInMiles' and is indexed by 'trip_id'
                    return pd.DataFrame(columns=['distanceInMiles'], index=pd.Index([], name='trip_id'))

                # Assuming trip_pmt has relevant columns, e.g., 'distanceInMiles'
                # Select relevant columns and return
                # Need to confirm actual columns available in BeamRunOutputData.trip_pmt
                # For now, returning the whole dataframe as a placeholder
                return trips_df[['distanceInMiles']] # Example: select a relevant column
            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.trip_pmt data: {e}")
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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM trips DataFrame
        from a single ModelOutputData instance.
        """
        def beam_trips_by_year_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            try:
                 return outputData.trips.dataFrame # Access BEAM trips dataFrame
            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.trips data: {e}")
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

    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Return the accessor function that extracts the BEAM skims DataFrame
        from a single ModelOutputData instance.
        """
        def beam_skims_by_year_accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            try:
                 return outputData.skims.dataFrame # Access BEAM skims dataFrame
            except Exception as e:
                logger.error(f"Error accessing BeamRunOutputData.skims data: {e}")
                return None
        return beam_skims_by_year_accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        # Assuming BEAM skims data is indexed by origin, destination, mode, etc.
        # This needs to be confirmed from the actual data structure.
        return ["origin", "destination", "mode", "time_period"] # Placeholder, needs verification
