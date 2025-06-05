from typing import Dict, Tuple, Optional

import pandas as pd

from src.geometry import Geometry
from src.input_directories import PilatesRunInputDirectory
from src.processed_data_frame import InfoByYear, TAZBasedDataFrame, InfoByIteration


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
