import hashlib
import os
import pathlib
import subprocess
from gzip import BadGzipFile
from io import BytesIO, StringIO
from typing import Iterable, Optional, Dict
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

# Use requests for more flexible HTTP calls (e.g., HEAD)
import requests
from google.cloud import storage
from typing import Union, List, Tuple, Any


from tables import HDF5ExtError
from tqdm import tqdm
import shutil

import pandas as pd
import numpy as np
import geopandas as gpd
import openmatrix as omx
import urllib.request  # Still useful for direct downloads

# --- Constants ---
TMP_DIR = ".tmp"
OMX_FILE_NAME = "skims.omx"


class InputDirectory:
    """
    Represents an input directory of raw data for postprocessing

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
        inputDirectory (InputDirectory): The parent input directory.
        index_col: Optional parameter for specifying the column(s) to use as the row labels.
        dtype: Optional parameter for specifying column data types.
        _file: Internal variable to store the loaded file DataFrame.
    """

    def __init__(
        self,
        inputDirectory: InputDirectory,
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
                    if self.filePath.endswith(".parquet"):
                        file = pd.read_parquet(self.filePath)
                        self._file = file
                    elif self.filePath.endswith(".csv") | self.filePath.endswith(
                        ".csv.gz"
                    ):
                        self._file = pd.read_csv(
                            self.filePath,
                            index_col=self.index_col,
                            dtype=self.dtype,
                        )

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


class Geometry:
    def __init__(self):
        self.region = None
        self.crs = None
        self._gdf = None
        self.unit = None
        self._path = None
        self._inputcrs = None
        self._gdf = None
        self.index = None
        self._otherFiles = dict()

    @property
    def gdf(self):
        if self._gdf is None:
            self.load()
        return self._gdf

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )

    def zoneToCountyMap(self):
        return NotImplementedError("This region is not defined yet")


class SfBayGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "SFBay"
        self.crs = "epsg:26910"
        self.unit = "TAZ"
        self.index = "taz1454"
        self._path = "geoms/sfbay-tazs-epsg-26910.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        return self._gdf.set_index(self.index)["areatype10"].to_dict()


class AustinGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Austin"
        self.crs = "epsg:26910"
        self.unit = "BG"
        self.index = "TAZ"
        self._path = "geoms/block_group_austin_26910.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Austin")


class SeattleGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Seattle"
        self.crs = "epsg:32048"
        self.unit = "BG"
        self.index = "OBJECTID"
        self._path = "geoms/block-groups-32048.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Seattle")

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )
        self._gdf.rename(columns={"county_nam": "county"}, inplace=True)


class EventsFile(RawOutputFile):
    """
    Represents an events file produced by BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(
        self, inputDirectory: InputDirectory, iteration: int, file_format="csv"
    ):
        """
        Initializes an EventsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The input directory where the file is stored.
            iteration (int): The BEAM iteration number to use.
        """
        dtypes = {
            "type": str,
            "numPassengers": "Int64",
            "driver": "str",
            "riders": "str",
            "linkTravelTime": "str",
            "links": "str",
            "person": "str",
            "vehicle": "str",
            "parkingTaz": "str",
        }
        if file_format == "csv":
            filename = "{0}.events.csv.gz".format(iteration)
        elif file_format == "parquet":
            filename = "{0}.events.parquet".format(iteration)
        else:
            raise ValueError("Unsupported file format: {0}".format(file_format))
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            filename,
        ]
        super().__init__(inputDirectory, relativePath, dtype=dtypes)
        self.eventTypes = dict()
        self.__file_format = file_format
        self.__chunksize = 5000000

    def collectEvents(self, eventTypes: list):
        """
        Collects specific event types from the raw events file and stores them in the EventsFile instance.

        Parameters:
            eventTypes (list): A list of event types to collect from the raw events file.
        """
        __listOfFrames = {eventType: [] for eventType in eventTypes}
        if self.__file_format == "csv":
            for chunk in pd.read_csv(
                self.filePath,
                chunksize=self.__chunksize,
                dtype={
                    "driver": "str",
                    "riders": "str",
                    "linkTravelTime": "str",
                    "links": "str",
                    "person": "str",
                    "vehicle": "str",
                    "parkingTaz": "str",
                },
            ):
                for eventType in eventTypes:
                    __listOfFrames[eventType].append(
                        chunk.loc[chunk["type"] == eventType, :].dropna(
                            axis=1, how="all"
                        )
                    )
        elif self.__file_format == "parquet":
            for eventType in eventTypes:
                __listOfFrames[eventType] = [
                    pd.read_parquet(
                        self.filePath, filters=[("type", "==", eventType)]
                    ).dropna(how="all", axis=1)
                ]
        for eventType in eventTypes:
            print("Extracting {0} events from raw events file".format(eventType))
            self.eventTypes[eventType] = pd.concat(
                __listOfFrames.pop(eventType), axis=0
            )

    def clearEvents(self):
        print("Clearing events memory")
        self.eventTypes.clear()


class LinkStatsFile(RawOutputFile):
    """
    Represents a linkStats file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory, iteration: int):
        """
        Initializes a LinkStatsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
            iteration (int): The iteration number.
        """
        relativePath = [
            "ITERS",
            "it.{0}".format(iteration),
            "{0}.linkstats.csv.gz".format(iteration),
        ]
        super().__init__(inputDirectory, relativePath, index_col=["link", "hour"])
        self.iteration = iteration


class NetworkFile(RawOutputFile):
    """
    Represents a network file from BEAM.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory, geometry: Geometry):
        """
        Initializes a Network instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
            :param geometry:
        """
        relativePath = "network.csv.gz"
        super().__init__(inputDirectory, relativePath, index_col="linkId")
        self.crs = geometry.crs


class InputPlansFile(RawOutputFile):
    """
    Represents an input plans file generated for BEAM by ActivitySim.

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "plans.csv.gz"
        super().__init__(inputDirectory, relativePath)


class ReplanningEventReasonFile(RawOutputFile):
    """
    Represents all replanning events in BEAM

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "replanningEventReason.csv"
        super().__init__(inputDirectory, relativePath)


class ScoreStatsFile(RawOutputFile):
    """
    Keeps track of agent scores in a BEAM run

    Attributes:
        (inherits attributes from OutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes an InputPlansFile instance.

        Parameters:
            inputDirectory (InputDirectory): The output directory where the file will be stored.
        """
        relativePath = "scorestats.txt"
        super().__init__(inputDirectory, relativePath)

    def file(self):
        """
        Property to lazily load the file and return it.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        if self._file is None:
            print("Reading file from {0}".format(self.filePath))
            try:
                self._file = pd.read_table(
                    self.filePath, index_col=self.index_col, dtype=None
                )
                if self._file.columns[0].startswith("<!"):
                    raise pd.errors.ParserError
            except FileNotFoundError:
                print("File at {0} does not exist".format(self.filePath))
                return None
            except pd.errors.ParserError:
                print("Initial download failed")
                bucket = storage.Client().get_bucket(self.filePath.split("/")[3])
                blob = bucket.get_blob("/".join(self.filePath.split("/")[4:]))
                if blob is not None:
                    fmt = ".txt"
                    path = pathlib.Path.cwd().joinpath(".tmp", self.hash() + fmt)
                    print("Downloading file from gcloud to {0}".format(path))
                    blob.download_to_filename(path)

                    self._file = pd.read_table(
                        path, index_col=self.index_col, dtype=None
                    )
                    print("Success! Deleting temporary file")
                    os.remove(path)
                else:
                    print("Giving up!")
                    return None
        return self._file


class BeamRunInputDirectory(InputDirectory):
    """
    Represents an input directory specific to a BEAM run.

    Attributes:
        eventsFile (EventsFile): The events file for the Beam run.
        inputPlansFile (InputPlansFile): The input plans file for the Beam run.
        linkStatsFile (LinkStatsFile): The link stats file for the Beam run.
        (inherits attributes from InputDirectory)
    """

    def __init__(
        self,
        baseFolderName: str,
        numberOfIterations: int = 0,
        geometry: Optional[Geometry] = None,
        region: Optional[str] = None,
        file_format: Optional[str] = "csv",
    ):
        """
        Initializes a BeamRunInputDirectory instance.

        Parameters:
            baseFolderName (str): The base folder name for the Beam run.
            numberOfIterations (int): The number of iterations for the Beam run.
        """
        super().__init__(baseFolderName, file_format)
        self.eventsFile = EventsFile(self, numberOfIterations, file_format)
        self.numberOfIterations = numberOfIterations
        self.inputPlansFile = InputPlansFile(self)
        self.replanningEventReasonFile = ReplanningEventReasonFile(self)
        self.scoreStatsFile = ScoreStatsFile(self)
        self.__linkStatsFile = {
            numberOfIterations: LinkStatsFile(self, numberOfIterations)
        }
        if (region is not None) & (geometry is None):
            if region == "SFBay":
                self.geometry = SfBayGeometry(
                    otherFiles={
                        "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
                    }
                )
            elif region == "Austin":
                self.geometry = AustinGeometry(otherFiles=dict())
            elif region == "Seattle":
                self.geometry = SeattleGeometry(otherFiles=dict())
            else:
                self.geometry = Geometry()
        else:
            self.geometry = geometry
        self.networkFile = NetworkFile(self, self.geometry)

    def linkStatsFile(self, numberOfIterations: Optional[int] = None):
        if numberOfIterations is None:
            numberOfIterations = self.numberOfIterations
        if numberOfIterations not in self.__linkStatsFile:
            self.__linkStatsFile[numberOfIterations] = LinkStatsFile(
                self, numberOfIterations
            )
        return self.__linkStatsFile[numberOfIterations]


class TripUtilitiesFiles(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        relativePath = "trip_mode_choice.zip"
        super().__init__(inputDirectory, relativePath, index_col="person_id")
        # self._file = None # Handled by base class

    def __hash(self):
        """
        Generates a hash based on the input directory path and class name.
        Used for the temporary folder name for extracted zip contents.

        Returns:
            str: The generated hash.
        """
        m = hashlib.md5()
        # Include the input directory path and the class name
        m.update(self.inputDirectory.directoryPath.encode())
        m.update(self.__class__.__name__.encode())
        return m.hexdigest()

    def file(self):
        """
        Property to lazily load the trip mode choice utilities data from a zip file.
        Downloads the zip file to a temporary location if it's remote and not cached.
        Extracts utilities.csv files from subfolders within the zip and concatenates them.

        Returns:
            pd.DataFrame: The loaded and concatenated utilities DataFrame, or None if loading fails.
        """
        if self._file is None:
            print(f"Attempting to load trip utilities from {self.filePath}")
            out = dict()
            # Temporary folder for extracted zip contents
            folderName = os.path.join(TMP_DIR, self.__hash())

            # Ensure the temporary directory exists
            os.makedirs(TMP_DIR, exist_ok=True)

            # Check if the extracted folder already exists (cached extraction)
            # Check for a sentinel file or just the presence of the main subdir
            sentinel_file = os.path.join(folderName, "extraction_complete.sentinel")
            if not os.path.exists(folderName) or not os.path.exists(sentinel_file):
                print(
                    f"Temporary extraction folder {folderName} not found or incomplete. Attempting download and extraction."
                )
                # Clean up potentially incomplete folder before re-extracting
                if os.path.exists(folderName):
                    print(f"Cleaning up incomplete extraction folder {folderName}")
                    shutil.rmtree(folderName)
                os.makedirs(folderName)  # Recreate empty folder

                try:
                    print(f"Downloading zip file from {self.filePath}")
                    # Download the zip file to a temporary file first
                    zip_temp_path = os.path.join(
                        TMP_DIR, f"{self.hash()}_utilities.zip"
                    )
                    urllib.request.urlretrieve(self.filePath, zip_temp_path)
                    print(f"Downloaded zip file to {zip_temp_path}")

                    # Extract the zip file
                    print(f"Extracting zip file to {folderName}")
                    with ZipFile(zip_temp_path, "r") as zfile:
                        # Extract only the utility files
                        utilities_files = [
                            f
                            for f in zfile.filelist
                            if f.filename.endswith("utilities.csv")
                        ]
                        for ls in tqdm(utilities_files, desc="Extracting utilities"):
                            # Construct target path within the temporary folder
                            target_path = os.path.join(folderName, ls.filename)
                            # Ensure parent directory exists
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            # Extract the file
                            zfile.extract(
                                ls.filename, folderName
                            )  # Extract relative to folderName

                    # Create a sentinel file to mark successful extraction
                    with open(sentinel_file, "w") as f:
                        f.write("Extraction complete")

                    print("Extraction successful.")
                    # Clean up the temporary zip file
                    # os.remove(zip_temp_path) # Commented out for debugging, enable in production

                except (FileNotFoundError, HTTPError, URLError) as e:
                    print(f"Failed to download zip file from {self.filePath}: {e}")
                    # Clean up the incomplete extraction folder
                    if os.path.exists(folderName):
                        shutil.rmtree(folderName)
                    return None  # Cannot proceed if download fails
                except Exception as e:
                    print(f"An unexpected error occurred during zip processing: {e}")
                    if os.path.exists(folderName):
                        shutil.rmtree(folderName)
                    return None

            # Now read the extracted files from the temporary folder
            print(f"Reading extracted utility files from {folderName}")
            extracted_utilities_dir = os.path.join(
                folderName, "trip_mode_choice"
            )  # Assuming this subdirectory exists within the zip
            if not os.path.exists(extracted_utilities_dir):
                print(
                    f"Error: Expected subdirectory '{extracted_utilities_dir}' not found after extraction."
                )
                return None

            try:
                files = os.listdir(extracted_utilities_dir)
                for file in tqdm(files, desc="Reading utility CSVs"):
                    if file.endswith("utilities.csv"):
                        # file is like '0_trip_mode_choice_utilities.csv'
                        # groupName should be the division ID, which is the number before '_trip_mode_choice_utilities.csv'
                        try:
                            groupName_str = file.split("_")[0]
                            groupName = int(groupName_str)  # Convert to int
                        except (IndexError, ValueError):
                            print(
                                f"Warning: Could not parse division ID from filename: {file}. Skipping."
                            )
                            continue

                        file_path_full = os.path.join(extracted_utilities_dir, file)
                        try:
                            df = pd.read_csv(
                                file_path_full,
                                index_col="trip_id",
                                # dtype=... # Add dtype if needed
                            )
                            out[groupName] = df
                        except Exception as read_e:
                            print(
                                f"Error reading extracted file {file_path_full}: {read_e}. Skipping."
                            )

                if out:
                    # Concatenate all the dataframes from the dictionary
                    self._file = pd.concat(out, names=["division", "trip_id"])
                    print("Successfully concatenated utility files.")
                else:
                    print("No utility files successfully read.")
                    self._file = None

            except FileNotFoundError:
                print(
                    f"Error: Could not list files in extracted directory {extracted_utilities_dir}."
                )
                self._file = None
            except Exception as e:
                print(
                    f"An unexpected error occurred while reading extracted files: {e}"
                )
                self._file = None

        return self._file


class PersonsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        if inputDirectory.file_format == "csv":
            relativePath = "persons.csv.gz"
        else:
            relativePath = "persons"
        super().__init__(inputDirectory, relativePath, index_col="person_id")

    def split(self, person_id_to_division) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        households = self.file()["household_id"].reset_index()
        households["division"] = households["person_id"].map(person_id_to_division)
        household_id_to_division = households.set_index("person_id")[
            "division"
        ].to_dict()
        gb = self.file().groupby(person_id_to_division)
        division_to_persons = {f: d for f, d in gb}
        return division_to_persons, household_id_to_division


class HouseholdsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        if inputDirectory.file_format == "csv":
            relativePath = "households.csv.gz"
        else:
            relativePath = "households"
        super().__init__(inputDirectory, relativePath, index_col="household_id")

    def split(
        self, household_id_to_division
    ) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        gb = self.file().groupby(household_id_to_division)
        division_to_households = {f: d for f, d in gb}
        return division_to_households, dict()


class TripsFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        if inputDirectory.file_format == "csv":
            relativePath = "final_trips.csv.gz"
        else:
            relativePath = "trips"
        super().__init__(
            inputDirectory,
            relativePath,
            index_col="trip_id",
            dtype={"household_id": int, "person_id": int, "tour_id": int},
        )

    def split(self, trip_id_to_division) -> (Dict[str, pd.DataFrame], Dict[int, str]):
        persons = self.file()["person_id"].reset_index()
        persons["division"] = persons["trip_id"].map(trip_id_to_division)

        divToPersons = persons.groupby("division").agg({"person_id": "unique"})
        tempMap = dict()
        tempMap[divToPersons.iloc[0].name] = set(divToPersons.iloc[0].person_id)
        for div, ps in divToPersons.iterrows():
            setPersons = set(ps.person_id)
            unassigned = True
            for finalDiv, finalPersons in tempMap.items():
                if not setPersons.isdisjoint(finalPersons):
                    finalPersons.update(setPersons)
                    unassigned = False
            if unassigned:
                tempMap[div] = setPersons

        outputMap = dict()
        outputMap[divToPersons.iloc[0].name] = tempMap[divToPersons.iloc[0].name]
        for div, setPersons in tempMap.items():
            unassigned = True
            for finalDiv, finalPersons in outputMap.items():
                if not setPersons.isdisjoint(finalPersons):
                    finalPersons.update(setPersons)
                    unassigned = False
            if unassigned:
                outputMap[div] = setPersons

        person_id_to_division = (
            pd.concat(
                [
                    pd.MultiIndex.from_product(
                        [[a], list(b)], names=["division", "person_id"]
                    ).to_frame(False)
                    for a, b in outputMap.items()
                ]
            )
            .set_index("person_id")
            .to_dict()["division"]
        )

        trips = self.file()
        trips["division_new"] = trips["person_id"].map(person_id_to_division)
        gb = trips.groupby("division_new")
        division_to_trips = {f: d for f, d in gb}
        trip_id_to_division_new = trips["division_new"].to_dict()
        return division_to_trips, person_id_to_division, trip_id_to_division_new


class ToursFile(RawOutputFile):
    def __init__(self, inputDirectory: InputDirectory):
        if inputDirectory.file_format == "csv":
            relativePath = "final_tours.csv.gz"
        else:
            relativePath = "tours"
        super().__init__(
            inputDirectory,
            relativePath,
            index_col="tour_id",
            dtype={"household_id": int, "person_id": int, "trip_id": int},
        )


class SkimsFile(RawOutputFile):
    """
    Represents a skims file used in activity-based models.
    Handles loading OMX files from local or remote paths.

    Attributes:
        (inherits attributes from RawOutputFile)
    """

    def __init__(self, inputDirectory: InputDirectory):
        """
        Initializes a SkimsFile instance.

        Parameters:
            inputDirectory (InputDirectory): The directory where the skims file is located.
        """
        # Relative path to the OMX file within the input directory
        # Assuming it's in activitysim/data or activitysim/data/data
        # Try the more common path first
        relativePath = ["activitysim", "data", OMX_FILE_NAME]
        filePathGuess = inputDirectory.append(relativePath)

        # Temporary location for the downloaded OMX file
        omx_temp_loc = os.path.join(
            TMP_DIR, f"{self.hash(inputDirectory)}_{OMX_FILE_NAME}"
        )

        # Use the temp location as the effective file path for the base class,
        # even though we handle the download/loading logic here.
        # This hash needs to be consistent regardless of whether the file is local or remote.
        # The file method will handle the download to this temp location if it doesn't exist.
        super().__init__(
            inputDirectory, relativePath=relativePath, file=None
        )  # filePath is set by base class

        # We need to override the file() method because it's not a simple CSV.
        # Store the calculated temporary location
        self._omx_temp_loc = omx_temp_loc
        # Store the original relative path attempt
        self._relativePathAttempt = relativePath

    def file(self):
        """
        Property to lazily load the skims data from an OMX file.
        Downloads the OMX file to a temporary location if it's remote and not cached.
        Extracts relevant skims (DistanceMiles, transit times, drive time, walk time).
        Returns a DataFrame indexed by ('Origin', 'Destination').

        Returns:
            pd.DataFrame: The loaded and processed skims DataFrame, or None if loading fails.
        """
        if self._file is None:
            print(f"Attempting to load skims from {self.filePath}")

            # Ensure the temporary directory exists
            os.makedirs(TMP_DIR, exist_ok=True)

            # Check if the temporary OMX file already exists (cached download)
            if not os.path.exists(self._omx_temp_loc):
                print(
                    f"Temporary OMX file not found at {self._omx_temp_loc}. Attempting download."
                )
                try:
                    # Try downloading the file. Check both common paths.
                    try:
                        url_to_download = self.inputDirectory.append(
                            self._relativePathAttempt
                        )
                        urllib.request.urlretrieve(url_to_download, self._omx_temp_loc)
                        print(f"Downloaded from {url_to_download}")
                    except Exception as e1:
                        print(f"Attempt 1 failed: {e1}. Trying alternative path.")
                        # Try the alternative path: activitysim/data/data/skims.omx
                        alt_relativePath = [
                            "activitysim",
                            "data",
                            "data",
                            OMX_FILE_NAME,
                        ]
                        url_to_download = self.inputDirectory.append(alt_relativePath)
                        urllib.request.urlretrieve(url_to_download, self._omx_temp_loc)
                        print(f"Downloaded from {url_to_download}")

                except Exception as download_e:
                    print(
                        f"Failed to download OMX file from {self.filePath} or alternative paths: {download_e}"
                    )
                    return None  # Cannot proceed if download fails

            # Now that the OMX file is confirmed to be at _omx_temp_loc, open it
            try:
                print(f"Opening OMX file from {self._omx_temp_loc}")
                sk = omx.open_file(self._omx_temp_loc, "r")

                # Get the shape of the matrices (assuming they are consistent)
                # Use a known matrix name like 'SOV_DIST__AM' to get dimensions
                if "SOV_DIST__AM" not in sk.list_matrices():
                    print(
                        f"Error: 'SOV_DIST__AM' matrix not found in {self._omx_temp_loc}"
                    )
                    sk.close()
                    return None

                matrix_shape = sk["SOV_DIST__AM"].shape
                num_zones = matrix_shape[0]
                zone_index = pd.Index(
                    np.arange(1, num_zones + 1)
                )  # Assuming zones are 1-indexed

                # Initialize a list to hold dataframes for concatenation
                dfs_to_concat = []

                # Extract DistanceMiles
                if "SOV_DIST__AM" in sk.list_matrices():
                    distMat = np.array(sk["SOV_DIST__AM"])
                    distDf = (
                        pd.DataFrame(distMat, index=zone_index, columns=zone_index)
                        .stack()
                        .rename("DistanceMiles")
                    )
                    dfs_to_concat.append(distDf)
                else:
                    print("Warning: SOV_DIST__AM matrix not found.")

                # Extract Transit Times (Iterate through common transit modes)
                transitModes = ["COM", "LOC", "HVY", "LRF"]
                for m in transitModes:
                    ivt_matrix_name = f"WLK_{m}_WLK_TOTIVT__AM"  # Total In-Vehicle Time
                    wait_matrix_name_iwait = f"WLK_{m}_WLK_IWAIT__AM"  # Initial Wait
                    wait_matrix_name_xwait = f"WLK_{m}_WLK_XWAIT__AM"  # Transfer Wait
                    wait_matrix_name_wacc = f"WLK_{m}_WLK_WACC__AM"  # Walk to Access
                    wait_matrix_name_wegr = f"WLK_{m}_WLK_WEGR__AM"  # Walk from Egress
                    wait_matrix_name_waux = f"WLK_{m}_WLK_WAUX__AM"  # Walk Auxiliary (might be in-vehicle walk?)

                    # Check if necessary matrices exist
                    if ivt_matrix_name in sk.list_matrices():
                        transitTimeMat = np.array(sk[ivt_matrix_name])
                        # Convert time from hundreths of minutes to hours
                        transitTimeDf = (
                            pd.DataFrame(
                                transitTimeMat / 100.0 / 60.0,
                                index=zone_index,
                                columns=zone_index,
                            )
                            .stack()
                            .rename(f"transitTravelTimeHours_{m}")
                        )
                        dfs_to_concat.append(transitTimeDf)
                    else:
                        print(f"Warning: {ivt_matrix_name} matrix not found.")

                    # Sum up wait times (convert from hundreths of minutes to hours)
                    wait_matrices = [
                        m
                        for m in [
                            wait_matrix_name_iwait,
                            wait_matrix_name_xwait,
                            wait_matrix_name_wacc,
                            wait_matrix_name_wegr,
                            wait_matrix_name_waux,
                        ]
                        if m in sk.list_matrices()
                    ]
                    if wait_matrices:
                        total_wait_mat = np.zeros(matrix_shape)
                        for wm_name in wait_matrices:
                            total_wait_mat += np.array(sk[wm_name])
                        transitWaitDf = (
                            pd.DataFrame(
                                total_wait_mat / 100.0 / 60.0,
                                index=zone_index,
                                columns=zone_index,
                            )
                            .stack()
                            .rename(f"transitWaitTimeHours_{m}")
                        )
                        dfs_to_concat.append(transitWaitDf)
                    elif any(m.startswith(f"WLK_{m}_") for m in sk.list_matrices()):
                        print(
                            f"Warning: Some {m} wait time matrices found, but not all common ones. Check skims contents."
                        )
                    else:
                        print(f"Warning: No wait time matrices found for mode {m}.")

                # Extract Drive Time (convert from hundreths of minutes to hours)
                if "SOV_TIME__AM" in sk.list_matrices():
                    driveTimeMat = np.array(sk["SOV_TIME__AM"])
                    driveTimeDf = (
                        pd.DataFrame(
                            driveTimeMat / 100.0 / 60.0,
                            index=zone_index,
                            columns=zone_index,
                        )
                        .stack()
                        .rename("driveTimeHours")
                    )
                    dfs_to_concat.append(driveTimeDf)
                else:
                    print("Warning: SOV_TIME__AM matrix not found.")

                # Calculate Walk Time (assuming 2.5 mph)
                # Need DistanceMiles for this calculation
                if "DistanceMiles" in [df.name for df in dfs_to_concat]:
                    distance_series = [
                        df for df in dfs_to_concat if df.name == "DistanceMiles"
                    ][0]
                    walkTimeHoursSeries = (distance_series / 2.5).rename(
                        "walkTimeHours"
                    )
                    dfs_to_concat.append(walkTimeHoursSeries)
                else:
                    print(
                        "Warning: DistanceMiles not available. Cannot calculate walk time."
                    )

                # Concatenate all extracted series into a single DataFrame
                # The index will be a MultiIndex ('Origin', 'Destination') after stacking.
                if dfs_to_concat:
                    distDf = pd.concat(dfs_to_concat, axis=1)
                    self._file = distDf
                    print("Successfully loaded and processed skims data.")
                else:
                    print("No skim matrices found to process.")
                    self._file = None

                sk.close()  # Close the OMX file

            except FileNotFoundError:
                print(
                    f"Error: OMX temporary file not found at {self._omx_temp_loc} after attempting download."
                )
                self._file = None
            except HDF5ExtError:
                print(
                    f"Error: Could not open OMX file at {self._omx_temp_loc}. It might be corrupted or not a valid OMX."
                )
                # Consider deleting the corrupted temp file here so it's re-downloaded next time
                if os.path.exists(self._omx_temp_loc):
                    try:
                        os.remove(self._omx_temp_loc)
                        print(
                            f"Removed potentially corrupted temporary file: {self._omx_temp_loc}"
                        )
                    except Exception as cleanup_e:
                        print(
                            f"Error cleaning up temporary file {self._omx_temp_loc}: {cleanup_e}"
                        )
                self._file = None
            except Exception as e:
                print(f"An unexpected error occurred while processing OMX file: {e}")
                self._file = None

        return self._file

    def hash(self, inputDirectory=None):
        """
        Generates a hash for the SkimsFile based on the input directory path
        and the relative path to the skims file. This is used for the temporary
        OMX file name.

        Returns:
            str: The generated hash.
        """
        if inputDirectory is None:
            inputDirectory = self.inputDirectory
        m = hashlib.md5()
        # Use the input directory path and the *intended* relative path for hashing,
        # regardless of which specific path variant was successfully downloaded.
        # Using self._relativePathAttempt ensures consistency.
        m.update(inputDirectory.directoryPath.encode())
        m.update(str(self._relativePathAttempt).encode())  # Convert list/str to bytes
        return m.hexdigest()


class ActivitySimRunInputDirectory(InputDirectory):
    def __init__(self, baseFolderName: str, geometry=Geometry(), file_format="csv"):
        super().__init__(baseFolderName, file_format)
        self.householdsFile = HouseholdsFile(self)
        self.personsFile = PersonsFile(self)
        self.tripsFile = TripsFile(self)
        self.toursFile = ToursFile(self)
        self.tripUtilitiesFiles = TripUtilitiesFiles(self)
        self.geometry = geometry

    def getSplitData(self):
        trip_id_to_division_raw = self.tripUtilitiesFiles.getInitialDivisionMapping()
        (
            division_to_trips,
            person_id_to_division,
            trip_id_to_division,
        ) = self.tripsFile.split(trip_id_to_division_raw)
        division_to_utilities, _ = self.tripUtilitiesFiles.split(trip_id_to_division)
        division_to_persons, household_id_to_division = self.personsFile.split(
            person_id_to_division
        )
        division_to_households, _ = self.householdsFile.split(household_id_to_division)
        return (
            division_to_utilities,
            division_to_trips,
            division_to_persons,
            division_to_households,
            person_id_to_division,
        )


class PilatesRunInputDirectory(InputDirectory):
    def __init__(
        self,
        baseFolderName: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations=0,
        region="SFBay",
        file_format="csv",
        collectEvents=False,
    ):
        super().__init__(baseFolderName, file_format)
        self.asimRuns = dict()
        self.beamRuns = dict()
        self.file_format = file_format
        try:
            self.skims = SkimsFile(self)
        except Exception as e:
            print("Skipping skims")
        if region == "SFBay":
            self.geometry = SfBayGeometry(
                otherFiles={
                    "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
                }
            )
        elif region == "Seattle":
            self.geometry = SeattleGeometry(otherFiles=dict())
        elif region == "Austin":
            self.geometry = AustinGeometry(otherFiles=dict())
        else:
            self.geometry = Geometry()
        for year in years:
            for asimLiteIteration in [-1, *np.arange(asimLiteIterations) + 1]:
                relPath = ["activitysim"]
                if not self.isLink:
                    relPath.append("output")
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                print("Loading year {0} it {1}".format(year, asimLiteIteration))
                self.asimRuns[(year, asimLiteIteration)] = ActivitySimRunInputDirectory(
                    self.append(relPath), self.geometry, file_format
                )
                relPath = ["beam"]
                if not self.isLink:
                    relPath.append("beam_output")
                    relPath.append(region.lower())
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                self.beamRuns[(year, asimLiteIteration)] = BeamRunInputDirectory(
                    self.append(relPath), beamIterations, self.geometry, file_format
                )
