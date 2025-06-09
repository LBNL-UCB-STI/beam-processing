from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Callable
import hashlib
import os
import pandas as pd

from src.processed_data_frame import ProcessedDataFrame
from src.input_base import OutputDirectory
from src.output_container import ModelOutputData
from .processed_data_frame_aggregators import InfoByYear, InfoByIteration  # Assuming these are in the same directory


class AggregatedProcessedDataFrameBase(ProcessedDataFrame, ABC):  # Inherit from ProcessedDataFrame and ABC
    """
    Base class for ProcessedDataFrames that aggregate data across multiple runs/iterations/years
    using an InfoByYear or InfoByIteration helper.
    Handles caching and the common loading pattern.
    Subclasses must define how to access the data for a single run.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: OutputDirectory,  # Expected to be a PilatesRunInputDirectory or similar
        *,  # Keyword-only arguments follow
        pilatesInputDict: dict,  # Explicitly require pilatesInputDict as keyword-only
        **kwargs,  # Other keyword arguments
    ):
        # pilatesInputDict is now a required keyword argument
        self._pilatesInputDict = pilatesInputDict

        self._inputDirectory = inputDirectory  # Store for accessors if needed

        # Calculate hash based on input directory path and the keys of the input dict
        m = hashlib.md5()
        m.update(str(inputDirectory.directoryPath).encode())
        m.update(self.__class__.__name__.encode())  # Include the specific class name
        # Sort keys for consistent hash across runs with same data but different dict order
        m.update(str(sorted(pilatesInputDict.keys())).encode())  # Use the required pilatesInputDict
        calculated_hash = m.hexdigest()

        # Pass relevant arguments to other superclasses
        # Assuming ProcessedDataFrame expects outputDataDirectory and inputDirectory
        # Assuming TAZBasedDataFrame expects pilatesInputDict
        # Passing all relevant ones using keyword arguments to ensure they are received correctly
        super().__init__(
            outputDataDirectory=outputDataDirectory,
            inputDirectory=inputDirectory,
            pilatesInputDict=pilatesInputDict,
            **kwargs,
        )

        self._accessor = self._get_accessor()
        self._expected_index_names = self._get_expected_index_names()
        self._cached_data = None

        # Pass calculated hash_key and other args/kwargs up to ProcessedDataFrame
        # indexedOn will be set by the load method

    @abstractmethod
    def _get_aggregator_type(self) -> Callable[..., InfoByYear | InfoByIteration]:
        """
        Abstract method: Return the type of aggregator helper (InfoByYear or InfoByIteration) to use.
        """
        pass

    @abstractmethod
    def _get_accessor(self) -> Callable[["ModelOutputData"], Optional[pd.DataFrame]]:
        """
        Abstract method: Return the accessor function that extracts the specific DataFrame
        from a single ModelOutputData instance (e.g., BeamRunOutputData, ActivitySimOutputData).
        """
        pass

    @abstractmethod
    def _get_expected_index_names(self) -> List[str]:
        """
        Abstract method: Return the expected index names of the DataFrame returned by the accessor.
        """
        pass

    def load(self) -> Optional[pd.DataFrame]:
        """
        Loads and aggregates data across runs using the specified aggregator helper and accessor.
        This implements the common loading logic for aggregated data products.
        """
        print(f"Loading {self.__class__.__name__} by aggregating across runs...")

        # Get the specific components from the abstract methods
        aggregator_type = self._get_aggregator_type()
        accessor_func = self._get_accessor()
        expected_names = self._get_expected_index_names()

        # Instantiate the chosen aggregator helper and run the aggregation
        aggregator_helper = aggregator_type(
            self._pilatesInputDict,
            accessor_func,
            expected_names,
            # Pass self._inputDirectory or other context if needed by helper's __init__
            # (InfoByYear/Iteration currently don't use inputDirectory in __init__,
            # but if they did, you might pass it here)
        )

        # Run the aggregation using the helper
        aggregated_df = aggregator_helper.run_aggregation()

        if aggregated_df is not None and not aggregated_df.empty:
            # Set the indexedOn attribute on this ProcessedDataFrame instance
            # The helper sets index names correctly, use them.
            self.indexedOn = list(aggregated_df.index.names)
            print(f"Finished aggregation for {self.__class__.__name__}.")
            return aggregated_df
        else:
            print(f"Aggregation for {self.__class__.__name__} returned no data.")
            # Return an empty DataFrame with the expected final index names (if possible)
            # The final index will be [year] + accessor_names for InfoByYear
            # or [year, iteration] + accessor_names for InfoByIteration
            if aggregator_type == InfoByYear:
                final_names = ["year"] + expected_names
            elif aggregator_type == InfoByIteration:
                final_names = ["year", "iteration"] + expected_names
            else:
                final_names = []  # Fallback
            self.indexedOn = final_names
            return pd.DataFrame(columns=[], index=pd.MultiIndex.from_tuples([], names=final_names))