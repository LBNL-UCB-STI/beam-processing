import hashlib
from typing import Dict, Tuple, Callable, Optional, List

import pandas as pd

from src.input_base import OutputDirectory
from src.processed_data_frame import ProcessedDataFrame


class InfoByYear(ProcessedDataFrame):
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
        outputDataDirectory: "OutputDataDirectory",  # Positional for OutputDataFrame
        inputDirectory: OutputDirectory,  # Positional for OutputDataFrame (expected to be PilatesRunInputDirectory)
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
        inputDirectory (OutputDirectory): The input directory (PilatesRunInputDirectory).
        pilatesInputDict (Dict[Tuple[int, int], "ModelOutputData"]): Dict of run data.
        accessor (Callable): Function to access data from a ModelOutputData instance.
        columns (List[str]): Expected index names of the accessor's DataFrame index.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",  # Positional for OutputDataFrame
        inputDirectory: OutputDirectory,  # Positional for OutputDataFrame (expected to be PilatesRunInputDirectory)
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
