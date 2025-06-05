import gc
import hashlib
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from src.constants import TMP_DIR
from src.input_base import OutputDirectory


# --- Constants ---


class ProcessedDataFrame(ABC):
    """
    Represents an output DataFrame with basic functionality like loading and preprocessing.
    Designed to load and preprocess raw BEAM outputs stored in an OutputDataDirectory.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (OutputDirectory): The input directory associated with the output.
        _dataFrame (pd.DataFrame): Internal variable to store the loaded DataFrame.
        _diskLocation (str): The file location for caching the DataFrame.
        indexedOn (str): The column to use as the index when loading data.
        _hash_key (Optional[str]): Optional custom hash key provided by subclass.
    """

    def __init__(
        self,
        outputDataDirectory: "OutputDataDirectory",
        inputDirectory: OutputDirectory,
        hash_key: Optional[str] = None,  # Made keyword-only
        *args,
        **kwargs,
    ):
        """
        Initializes an OutputDataFrame instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            inputDirectory (OutputDirectory): The associated input directory.
            hash_key (Optional[str]): A pre-calculated hash key from a subclass.
        """
        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        self._dataFrame = None
        self._hash_key = hash_key  # Store provided hash key

        # Use the provided hash_key if available, otherwise use default hash()
        calculated_hash = self._hash_key if self._hash_key is not None else self.hash()

        self._diskLocation = os.path.join(
            TMP_DIR,
            calculated_hash + ".parquet",
        )
        self.indexedOn = None
        # Pass remaining args/kwargs up the MRO chain
        super().__init__(*args, **kwargs)

    def hash(self):
        """
        Generates a hash based on the input and class name.
        Subclasses like InfoByYear should override this or provide hash_key in __init__.

        Returns:
            str: The generated hash.
        """
        # If a custom hash key was provided, use it
        if self._hash_key is not None:
            return self._hash_key

        # Otherwise, calculate default hash
        m = hashlib.md5()
        # Ensure inputDirectory.directoryPath exists and is a string
        input_path_str = str(getattr(self.inputDirectory, "directoryPath", ""))
        class_name_str = self.__class__.__name__

        m.update(input_path_str.encode())
        m.update(class_name_str.encode())
        return m.hexdigest()

    @property
    def cached(self) -> bool:
        """
        Checks if the DataFrame is cached.

        Returns:
            bool: True if the DataFrame is cached, False otherwise.
        """
        # Ensure temporary directory exists before checking cache file
        os.makedirs(TMP_DIR, exist_ok=True)
        return os.path.exists(self._diskLocation)

    def clearCache(self):
        """
        Clears the cached DataFrame.
        """
        if self.cached:
            try:
                os.remove(self._diskLocation)
                print(f"Cleared cache file: {self._diskLocation}")
            except OSError as e:
                print(f"Error clearing cache file {self._diskLocation}: {e}")
        self._dataFrame = None

    def clearMemory(self):
        if self._dataFrame is not None:
            print(f"Clearing memory for {self.__class__.__name__}!")
            del self._dataFrame
            self._dataFrame = None
            gc.collect()

    @abstractmethod
    def load(self):
        """
        Abstract method for loading data into a DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

    def _write(self, obj):
        assert isinstance(obj, pd.DataFrame)
        try:
            # Ensure temporary directory exists before writing
            os.makedirs(TMP_DIR, exist_ok=True)
            obj.to_parquet(self._diskLocation, engine="fastparquet")
        except Exception as e:
            print(
                f"Error writing {self.__class__.__name__} to {self._diskLocation}: {e}"
            )
            # Optionally remove incomplete file if writing fails
            if os.path.exists(self._diskLocation):
                try:
                    os.remove(self._diskLocation)
                except:
                    pass

    def _read(self):
        # Add error handling if tmp dir doesn't exist (unlikely but safe)
        if not os.path.exists(TMP_DIR):
            # This case should ideally be prevented by .cached check and _write logic,
            # but adding a check here for robustness.
            print(f"Temporary directory {TMP_DIR} does not exist during read attempt.")
            return None  # Indicate read failure

        if not os.path.exists(self._diskLocation):
            # This case might happen if cache was cleared between .cached check and _read,
            # or if .cached incorrectly returned True.
            print(
                f"Cache file does not exist at {self._diskLocation} during read attempt."
            )
            return None  # Indicate read failure

        try:
            return pd.read_parquet(self._diskLocation, engine="fastparquet")
        except Exception as e:
            print(
                f"Error reading {self.__class__.__name__} from {self._diskLocation}: {e}"
            )
            self.clearCache()
            return None  # Indicate read failure

    @property
    def dataFrame(self) -> Optional[pd.DataFrame]:
        """
        Property to lazily load the DataFrame and return it.
        Tries reading from cache first, then loads and caches if not available.

        Returns:
            pd.DataFrame: The loaded and preprocessed DataFrame, or None if loading/preprocessing fails.
        """
        if self._dataFrame is None:
            print(f"Accessing dataFrame for {self.__class__.__name__}...")
            if self.cached:
                print(
                    f"Reading {self.__class__.__name__} from cache: {self._diskLocation}"
                )
                self._dataFrame = self._read()
                # If read failed (e.g., corrupted cache), _dataFrame will still be None
                if self._dataFrame is not None:
                    print(f"Successfully read {self.__class__.__name__} from cache.")
            if self._dataFrame is None:
                print(
                    f"Cache miss or failed read for {self.__class__.__name__}. Loading raw data."
                )
                df = self.load()
                if df is None:
                    print(
                        f"Loading raw data for {self.__class__.__name__} returned None."
                    )
                    return None  # Loading failed

                print(
                    f"Preprocessing raw data for {self.__class__.__name__} ({df.shape[0]} rows)..."
                )
                self._dataFrame = self.preprocess(df)
                if self._dataFrame is None:
                    print(f"Preprocessing {self.__class__.__name__} returned None.")
                    return None  # Preprocessing failed

                print(
                    f"Preprocessing complete. Writing {self.__class__.__name__} ({self._dataFrame.shape[0]} rows) to cache: {self._diskLocation}"
                )
                self._write(self._dataFrame)
                # If writing failed, _dataFrame is still set, but the file might be incomplete/missing.
                # The read logic handles corrupted files by clearing cache.

        return self._dataFrame

    @dataFrame.setter
    def dataFrame(self, df: pd.DataFrame):
        """
        Setter for the dataFrame property.

        Parameters:
            df (pd.DataFrame): The DataFrame to set.
        """
        self._dataFrame = df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for preprocessing the DataFrame.
        Subclasses must implement this.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        print(
            f"No custom preprocessing defined for {self.__class__.__name__}. Returning raw data."
        )
        return df

    def toCsv(self):
        """
        Saves the DataFrame to a CSV file in the output data directory.
        """
        name = self.__class__.__name__
        output_dir = self.outputDataDirectory.path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, name + ".csv")
        print(f"Saving {name} to {output_path}")
        # Accessing .dataFrame triggers load/cache if needed
        df_to_save = self.dataFrame
        if df_to_save is not None and not df_to_save.empty:
            try:
                df_to_save.to_csv(output_path)
                print(f"Successfully saved {name} to CSV.")
            except Exception as e:
                print(f"Error saving {name} to {output_path}: {e}")
        else:
            print(
                f"No dataFrame available or dataFrame is empty for {name}. Not saving CSV."
            )

    def addMapping(self, mapping: dict, fromCol: str, toCol: str):
        """
        Adds a new column to the DataFrame based on a mapping.

        Parameters:
            mapping (dict): The mapping to use.
            fromCol (str): The source column in the index (level name).
            toCol (str): The new column to add.

        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        # Accessing dataFrame triggers load/cache
        df = self.dataFrame
        if df is not None:
            try:
                # Check if fromCol is in index levels
                if fromCol in df.index.names:
                    df[toCol] = df.index.get_level_values(fromCol).map(mapping)
                else:
                    print(
                        f"Warning: Column '{fromCol}' not found in DataFrame index levels."
                    )
                    # Optionally try mapping from a column if not in index
                    if fromCol in df.columns:
                        df[toCol] = df[fromCol].map(mapping)
                    else:
                        print(
                            f"Warning: Column '{fromCol}' not found in DataFrame columns either. Cannot add mapping."
                        )
                        # Add column with None or NaN?
                        df[toCol] = None  # Or np.nan or pd.NA depending on dtype
                return df
            except Exception as e:
                print(f"Error applying mapping to {self.__class__.__name__}: {e}")
                return df  # Return dataframe even on error
        return None  # Return None if dataFrame was None

    def unstackColumn(self, col, index):
        """
        Unstacks a specified column based on the provided index (level name).

        Parameters:
            col (str): The column to unstack.
            index: The level name in the index to unstack by.

        Returns:
            pd.DataFrame: The unstacked DataFrame, or None if dataFrame is None or unstacking fails.
        """
        df = self.dataFrame
        if df is not None:
            try:
                # Select the column first, then unstack
                if col in df.columns:
                    return df[col].unstack(index)
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame columns.")
                    return None
            except Exception as e:
                print(
                    f"Error unstacking column '{col}' by index '{index}' for {self.__class__.__name__}: {e}"
                )
                return None  # Indicate unstacking failure
        return None  # Return None if dataFrame was None

    def process(
        self,
        normalize: Optional[str],
        aggregateBy: Optional[List[str]],
        mapping: Optional[Dict[str, str]],
    ) -> pd.DataFrame:
        raise NotImplementedError("This class does not have process defined")


