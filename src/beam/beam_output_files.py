import hashlib
import os
import pathlib
import urllib.request

import numpy as np
import openmatrix as omx

import pandas as pd
from google.cloud import storage
from tables import HDF5ExtError

from src.geometry import Geometry
from src.input_base import RawOutputFile, InputDirectory
from src.constants import OMX_FILE_NAME, TMP_DIR


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
                        self.filePath,
                        filters=[("type", "==", eventType)],
                        dtype_backend="pyarrow",
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
