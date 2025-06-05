import hashlib
from typing import List, Optional, Dict, Tuple

import pandas as pd

from src.beam.beam_input_directory import BeamRunInputDirectory
from src.beam.beam_output_files import LinkStatsFile
from src.beam.beam_processed_data_frame import (
    LabeledNetwork,
    LinkStatsFromRawFile,
    NetworkVolumesByLink,
)
from src.constants import CONGESTION_THRESHOLD_MPH
from src.input_directories import PilatesRunInputDirectory
from src.processed_data_frame import (
    ProcessedDataFrame,
    InfoByIteration,
    InfoByYear,
    TAZBasedDataFrame,
)


class NetworkVolumesByLinkByIteration(ProcessedDataFrame):
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
            geoIndex=pilatesRunInputDirectory.geometry.index,
            pilatesInputDict=pilatesInputDict,
            accessor=accessor,
            columns=columns,
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
            geoIndex=pilatesRunInputDirectory.geometry.index,
            pilatesInputDict=pilatesInputDict,
            accessor=accessor,
            columns=columns,
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
