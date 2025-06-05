import hashlib
import os
import pathlib
from gzip import BadGzipFile
from typing import Union, List
from urllib.error import HTTPError, URLError

import pandas as pd
from google.cloud import storage

from src.constants import TMP_DIR


class OutputDirectory:
    """
    Represents directory of raw data for postprocessing

    Attributes:
        directoryPath (str): The path to the directory.
        isLink (bool): Indicates whether the directory path includes a link either to a website or an s3 or gcloud bucket
    """

    def __init__(self, path: str, file_format: str):
        self.directoryPath = path
        self.isLink = "://" in path
        self.file_format = file_format

    def append(self, relativePath):
        """
        Appends a relative path to the directory path and returns the combined path.

        Parameters:
            relativePath: The relative path to append.

        Returns:
            str: The combined path.
        """
        if self.isLink:
            if type(relativePath) in [list, tuple]:
                return "/".join([self.directoryPath] + relativePath)
            else:
                return "/".join([self.directoryPath, relativePath])
        else:
            if type(relativePath) in [list, tuple]:
                return os.path.join(self.directoryPath, *relativePath)
            elif type(relativePath) is str:
                return os.path.join(self.directoryPath, relativePath)


class RawOutputFile:
    """
    Represents a raw output file. It can be given additional optional properties like index_col and dtype.
    Handles loading from local files or cloud storage (GCS).

    Attributes:
        filePath (str): The path (local or URL) to the output file.
        inputDirectory (OutputDirectory): The parent input directory.
        index_col: Optional parameter for specifying the column(s) to use as the row labels.
        dtype: Optional parameter for specifying column data types.
        _file: Internal variable to store the loaded file DataFrame.
    """

    def __init__(
        self,
        inputDirectory: OutputDirectory,
        relativePath: Union[str, List[str]],  # Allow relativePath to be list or string
        index_col=None,
        dtype=None,
        file=None,  # Allow passing an already loaded DataFrame for internal use
    ):
        self.inputDirectory = inputDirectory
        # Append relative path to the base directory path
        self.filePath = self.inputDirectory.append(relativePath)
        self.index_col = index_col
        self.dtype = dtype
        self._file = file

    def file(self):
        """
        Property to lazily load the file and return it as a Pandas DataFrame.
        Handles local paths and GCS URLs. Includes basic error handling and
        attempts download from GCS if direct read fails (e.g., GZIP issues).

        Returns:
            pd.DataFrame: The loaded DataFrame, or None if loading fails.
        """
        if self._file is None:
            print(f"Attempting to read file from {self.filePath}")
            try:
                if self.inputDirectory.file_format == "csv":
                    # Try reading directly (works for local and some cloud/http paths)
                    # Add compression handling for gz
                    compression = "gzip" if self.filePath.endswith(".gz") else None
                    self._file = pd.read_csv(
                        self.filePath,
                        index_col=self.index_col,
                        dtype=self.dtype,  # Use provided dtype
                        compression=compression,
                    )
                    # Check if the file is an empty CSV or an HTML error page masquerading as CSV
                    if self._file.empty and self.filePath.endswith(".csv.gz"):
                        # Check if it's *just* headers, could indicate empty
                        # A more robust check might look at file size or content header if available
                        print(f"Warning: File {self.filePath} appears empty.")
                    elif self._file.columns.empty and self.filePath.endswith(".csv.gz"):
                        # Handle cases where read_csv might return empty columns for invalid gzip?
                        # Or if the file is truly empty.
                        print(f"Warning: File {self.filePath} has no columns.")
                        self._file = None  # Treat as failed load
                    elif self._file.columns[0].startswith("<!"):
                        # Catch reading an html file as a table
                        raise pd.errors.ParserError(
                            f"File {self.filePath} seems to be an HTML error page."
                        )
                elif self.inputDirectory.file_format == "parquet":
                    if self.filePath.endswith(".csv") | self.filePath.endswith(
                        ".csv.gz"
                    ):
                        self._file = pd.read_csv(
                            self.filePath,
                            index_col=self.index_col,
                            dtype=self.dtype,
                        )
                    else:
                        file = pd.read_parquet(self.filePath)
                        self._file = file

            except (
                FileNotFoundError,
                HTTPError,
                URLError,
                BadGzipFile,
                pd.errors.ParserError,
            ) as e:
                print(f"Initial read failed for {self.filePath}: {e}")

                # If it's a GCS path and read failed, try downloading locally as a fallback
                if self.inputDirectory.directoryPath.startswith("gs://"):
                    print(f"Attempting GCS download fallback for {self.filePath}")
                    try:
                        bucket_name = self.inputDirectory.directoryPath.split("/")[2]
                        # Construct blob path from the rest of the URL after bucket name
                        blob_path_parts = self.filePath.split("/")[3:]
                        # If relativePath was a list, reconstruct the blob path carefully
                        if isinstance(self.inputDirectory.append(relativePath), list):
                            blob_path_parts = self.filePath.split("/")[
                                3:
                            ]  # Assumes append logic joins correctly
                        blob_path = "/".join(blob_path_parts)

                        client = storage.Client()
                        bucket = client.get_bucket(bucket_name)
                        blob = bucket.blob(blob_path)

                        # Ensure the temporary directory exists
                        os.makedirs(TMP_DIR, exist_ok=True)

                        # Determine temporary file path
                        # Use the hash and original extension
                        file_extension = os.path.splitext(self.filePath)[-1]
                        if self.filePath.endswith(".csv.gz"):
                            temp_file_path = pathlib.Path(TMP_DIR).joinpath(
                                f"{self.hash()}.csv.gz"
                            )
                        elif self.filePath.endswith(".csv"):
                            temp_file_path = pathlib.Path(TMP_DIR).joinpath(
                                f"{self.hash()}.csv"
                            )
                        elif self.filePath.endswith(".txt"):
                            temp_file_path = pathlib.Path(TMP_DIR).joinpath(
                                f"{self.hash()}.txt"
                            )
                        # Add other formats as needed (e.g., .parquet, .zip, .omx - though OMX/ZIP handled separately)
                        else:
                            print(
                                f"Warning: Unknown file extension for temporary download: {self.filePath}"
                            )
                            temp_file_path = pathlib.Path(TMP_DIR).joinpath(
                                f"{self.hash()}_download"
                            )

                        print(
                            f"Downloading blob {blob_path} from bucket {bucket_name} to {temp_file_path}"
                        )
                        blob.download_to_filename(temp_file_path)

                        print("Download successful. Attempting to read temporary file.")
                        # Read from the downloaded temporary file
                        compression = (
                            "gzip" if str(temp_file_path).endswith(".gz") else None
                        )
                        self._file = pd.read_csv(
                            temp_file_path,
                            index_col=self.index_col,
                            dtype=self.dtype,
                            compression=compression,
                        )
                        print("Successfully read from temporary file.")

                        # Clean up the temporary file
                        # os.remove(temp_file_path) # Commented out for debugging, enable in production

                    except Exception as gcs_e:
                        print(
                            f"GCS download/read fallback failed for {self.filePath}: {gcs_e}"
                        )
                        self._file = None  # Ensure _file is None on failure

                else:
                    # If not a GCS path, and read failed, just set _file to None
                    self._file = None
                    print(f"Giving up on reading {self.filePath}.")

        # Ensure index is set correctly if index_col was specified and loading was successful
        if (
            self._file is not None
            and self.index_col is not None
            and not isinstance(self._file.index, pd.MultiIndex)
            and (
                isinstance(self.index_col, list)
                or self._file.index.name != self.index_col
            )
        ):
            # This might happen if index_col was specified but read_csv didn't apply it,
            # or if it was a list but the result wasn't a MultiIndex.
            # This could indicate an issue with the file or read_csv.
            # For robustness, let's try resetting and setting index explicitly if needed.
            try:
                if (
                    not isinstance(self.index_col, list)
                    and isinstance(self._file.columns, pd.Index)
                    and self.index_col in self._file.columns
                ):
                    self._file.set_index(self.index_col, inplace=True)
                elif isinstance(self.index_col, list) and all(
                    col in self._file.columns for col in self.index_col
                ):
                    self._file.set_index(self.index_col, inplace=True)
                # Add more robust checks if index is still wrong
            except KeyError as e:
                print(
                    f"Warning: Could not set specified index {self.index_col} on file {self.filePath}: {e}"
                )
            except Exception as e:
                print(
                    f"Warning: Unexpected error setting index {self.index_col} on file {self.filePath}: {e}"
                )

        return self._file

    def hash(self):
        """
        Generates a hash based on the input directory path and relative file path.
        This ensures cache files are unique per input run and file.

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        # Include both the base directory and the specific file path
        m.update(self.inputDirectory.directoryPath.encode())
        m.update(self.filePath.encode())  # Using self.filePath includes relative path
        return m.hexdigest()


def gcs_blob_exists(gcs_url: str) -> bool:
    """Checks if a blob exists at a given gs:// URL."""
    try:
        if not gcs_url.startswith("gs://"):
            # Not a GCS URL, return False or raise error depending on desired behavior
            print(f"Warning: gcs_blob_exists called with non-GCS URL: {gcs_url}")
            return False

        # Parse the bucket name and blob path from the URL
        parts = gcs_url.replace("gs://", "").split("/", 1)
        if len(parts) < 2:
            print(f"Warning: GCS URL '{gcs_url}' does not specify a blob path.")
            return False
        bucket_name = parts[0]
        blob_name = parts[1]

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()  # Use blob.exists() to check without downloading
    except Exception as e:
        print(f"Error checking GCS blob existence for '{gcs_url}': {e}")
        return False  # Assume not found or inaccessible on error
