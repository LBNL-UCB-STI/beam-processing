import hashlib
from typing import Dict, Tuple, Callable, Optional, List

import pandas as pd

from src.input_base import OutputDirectory
from src.processed_data_frame import ProcessedDataFrame


class InfoByYear:
    """
    Base class for output dataframes that aggregate data across years,
    typically using the last iteration for each year.

    Inherits from OutputDataFrame.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (OutputDirectory): The input directory (PilatesRunInputDirectory).
        pilatesInputDict (Dict[Tuple[int, int], "ModelOutputData"]): Dict of run data.
        accessor (Callable): Function to access data from a ModelOutputData instance.
        columns (List[str]): Expected index names of the accessor's output DataFrame.
        __lastIterationPerYear (Dict[int, int]): Mapping from year to the last iteration found.
    """

    def __init__(
        self,
        pilatesInputDict: Dict[Tuple[int, int], "ModelOutputData"],
        accessor: Callable[["ModelOutputData"], Optional[pd.DataFrame]],
        columns: List[str],
    ):
        self.pilatesInputDict = pilatesInputDict
        self.__accessor = accessor
        self.__columns = columns
        self.__lastIterationPerYear = {}
        if pilatesInputDict:
            for yr, it in pilatesInputDict.keys():
                if (
                    yr not in self.__lastIterationPerYear
                    or it >= self.__lastIterationPerYear[yr]
                ):
                    self.__lastIterationPerYear[yr] = it

    def run_aggregation(self):
        """
        Loads data from the last available iteration for each year using the accessor.
        Concatenates the results.
        """

        print(
            f"Loading data for {self.__class__.__name__} (last iteration per year)..."
        )
        data_frames = {}

        if self.__lastIterationPerYear:
            for yr in sorted(self.__lastIterationPerYear.keys()):
                data_frames[yr] = pd.DataFrame  # Init empty dataframe
                it = self.__lastIterationPerYear[yr]
                for current_it in range(yr, -2, -1):
                    print(f" - Attempting to load year {yr}, last iteration {it}")
                    # Try to load the data, iterate backward through iterations if the last one fails
                    if (yr, it) not in self.pilatesInputDict:
                        print(
                            f" - No data found for year {yr}, iteration {current_it}. Trying previous iterations."
                        )
                        break
                    data_instance = self.pilatesInputDict[(yr, it)]

                    # Access the data using the provided accessor
                    df = self.__accessor(data_instance, yr, it)
                    if not df is None:
                        if not df.empty:
                            print(
                                f" - Successfully loaded year {yr}, iteration {current_it} ({df.shape[0]} rows)"
                            )
                            data_frames[yr] = df
                            self.__lastIterationPerYear[yr] = current_it
                            break
                    if current_it < 0:
                        print(
                            f" - Could not load data for year {yr} after trying all iterations down to {current_it}."
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

        print(
            f"Finished loading {self.__class__.__name__}. Result shape: {combined_df.shape}"
        )

        return combined_df


class InfoByIteration:
    """

    Base class for output dataframes that aggregate data across iterations.
    Loads data from *all* available iterations for the specified year.

    Inherits from OutputDataFrame.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (OutputDirectory): The input directory (PilatesRunInputDirectory).
        pilatesInputDict (Dict[Tuple[int, int], "ModelOutputData"]): Dict of run data.
        accessor (Callable): Function to access data from a ModelOutputData instance.
        columns (List[str]): Expected index names of the accessor's DataFrame index.
    """

    def __init__(
        self,
        pilatesInputDict: Dict[Tuple[int, int], "ModelOutputData"],
        accessor: Callable[["ModelOutputData"], Optional[pd.DataFrame]],
        columns: List[str],
    ):
        self.pilatesInputDict = pilatesInputDict
        self.__accessor = accessor
        self.__columns = columns
        self.__lastIterationPerYear = {}
        if pilatesInputDict:
            for yr, it in pilatesInputDict.keys():
                if (
                    yr not in self.__lastIterationPerYear
                    or it >= self.__lastIterationPerYear[yr]
                ):
                    self.__lastIterationPerYear[yr] = it

    def run_aggregation(self):
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
                    # Pass year and iteration to the accessor
                    df = self.__accessor(data_instance, yr, it)
                    if df is not None and not df.empty:
                        data_frames[(yr, it)] = df
                        print(
                            f" - Loaded year {yr}, iteration {it} ({df.shape[0]} rows)"
                        )
                    else:
                        print(f" - No data for year {yr}, iteration {it}")
                except Exception as e:
                    print(f" - Error loading data for year {yr}, iteration {it}: {e}")

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
        print(
            f"Finished loading {self.__class__.__name__}. Result shape: {combined_df.shape}"
        )

        return combined_df
