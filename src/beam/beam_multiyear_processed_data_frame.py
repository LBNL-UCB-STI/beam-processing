from typing import List, Optional, Dict, Tuple, Callable

import pandas as pd

from src.constants import CONGESTION_THRESHOLD_MPH
from src.input_directories import PilatesRunOutputDirectory
from src.processed_data_frame_aggregators import InfoByYear, InfoByIteration
from src.processed_data_frame_mixins import TAZBasedDataFrame
from src.processed_dataframe_agg_base import AggregatedProcessedDataFrameBase


class ReplanningEventReasonByIteration(AggregatedProcessedDataFrameBase):
    """
    Aggregates replanning event reasons across multiple iterations using AggregatedProcessedDataFrameBase.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            df = outputData.replanningEventReasons.dataFrame

            if df is None or df.empty:
                return pd.DataFrame(
                    columns=["count"], index=pd.Index([], name="reason")
                )

            if "reason" in df.columns:
                return df["reason"].value_counts().to_frame("count")
            else:
                print(f"Warning: 'reason' column not found in replanningEventReasons dataFrame for iteration {iteration}. Attempting to use column names as reasons.")
                try:
                    df.columns.set_names("reason", inplace=True)
                    df = df.T
                    df.columns = ["count"]
                    return df
                except Exception as e:
                    print(f"Error processing replanningEventReasons for iteration {iteration} with column name fallback: {e}. Skipping iteration.")
                    return None

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["reason"]


class ScoreStatsByIteration(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.scoreStats.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return []


class ModeVMTByYear(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.modeVMT.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["mode_extended"]


class ModeEnergyByYear(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.modeEnergy.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["mode_extended"]


class ModePMTByYear(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.modePMT.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["mode_extended"]


class ModePMTByIteration(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.modePMT.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["mode_extended"]


class CongestionInfoByYear(TAZBasedDataFrame, AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict
        self.geometry = pilatesRunInputDirectory.geometry
        self.geoIndex = pilatesRunInputDirectory.geometry.index

        AggregatedProcessedDataFrameBase.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=pilatesInputDict
        )
        TAZBasedDataFrame.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            geometry=self.geometry,
            pilatesInputDict=self.pilatesInputDict,
            geoIndex=self.geoIndex,
        )

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
        def accessor(outputData: "BeamRunOutputData") -> Optional[pd.DataFrame]:
            tazIndex = self.geoIndex

            df = outputData.tazTrafficVolumes.dataFrame

            if df is None or df.empty:
                return pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([], names=["roadType", tazIndex])
                )

            df["mph"] = df["VMT"] / df["VHT"]
            df["congestedHours"] = (df["mph"] < CONGESTION_THRESHOLD_MPH).astype(
                int
            )

            df = df.groupby([tazIndex, "attributeOrigType"]).agg(
                {"VMT": "sum", "VHT": "sum", "congestedHours": "sum"}
            )
            df["mph"] = df["VMT"] / df["VHT"]

            df_unstacked = df.unstack(tazIndex)
            if isinstance(df_unstacked, pd.DataFrame):
                df_unstacked.columns.set_names("metric", level=0, inplace=True)
            elif isinstance(df_unstacked, pd.Series):
                df_unstacked = df_unstacked.index.set_names(
                    "metric", level=0
                ).to_frame()
            return df_unstacked

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["attributeOrigType"]


class CongestionInfoByIteration(TAZBasedDataFrame, AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict
        self.geometry = pilatesRunInputDirectory.geometry
        self.geoIndex = pilatesRunInputDirectory.geometry.index

        AggregatedProcessedDataFrameBase.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=self.pilatesInputDict,
        )
        TAZBasedDataFrame.__init__(
            self,
            outputDataDirectory,
            pilatesRunInputDirectory,
            geometry=self.geometry,
            geoIndex=self.geoIndex,
            pilatesInputDict=self.pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            tazIndex = self.geoIndex

            df = outputData.tazTrafficVolumes.dataFrame

            if df is None or df.empty:
                return pd.DataFrame(
                    index=pd.MultiIndex.from_tuples([], names=["roadType", tazIndex])
                )

            df["mph"] = df["VMT"] / df["VHT"]
            df["congestedHours"] = (df["mph"] < CONGESTION_THRESHOLD_MPH).astype(
                int
            )

            df = df.groupby([tazIndex, "attributeOrigType"]).agg(
                {"VMT": "sum", "VHT": "sum", "congestedHours": "sum"}
            )
            df["mph"] = df["VMT"] / df["VHT"]

            df_unstacked = df.unstack(tazIndex)
            if isinstance(df_unstacked, pd.DataFrame):
                df_unstacked.columns.set_names("metric", level=0, inplace=True)
            elif isinstance(df_unstacked, pd.Series):
                df_unstacked = df_unstacked.index.set_names(
                    "metric", level=0
                ).to_frame()
            return df_unstacked

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["attributeOrigType"]


class PassengerMilesByVehicleAndModeByYear(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=self.pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.passengerMilesByVehicleAndMode.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["vehicle_type", "mode_extended"]


class PassengerMilesByVehicleAndModeByIteration(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=self.pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.passengerMilesByVehicleAndMode.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["vehicle_type", "mode_extended"]


class RealizedModeCountByIteration(AggregatedProcessedDataFrameBase):

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        pilatesInputDict: Dict[Tuple[int, int], "BeamRunOutputData"],
    ):
        self.pilatesInputDict = pilatesInputDict

        super().__init__(
            outputDataDirectory,
            pilatesRunInputDirectory,
            pilatesInputDict=self.pilatesInputDict,
        )

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
        def accessor(outputData: "BeamRunOutputData", year: int, iteration: int) -> Optional[pd.DataFrame]:
            return outputData.realizedModeCount.dataFrame

        return accessor

    def _get_expected_index_names(self) -> List[str]:
        """
        Return the expected index names of the DataFrame returned by the accessor.
        """
        return ["mode_extended"]
