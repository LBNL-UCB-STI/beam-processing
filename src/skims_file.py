import hashlib
import os
import urllib.request

import numpy as np
import openmatrix as omx
import pandas as pd
from tables import HDF5ExtError

from src.constants import OMX_FILE_NAME, TMP_DIR
from src.input_base import RawOutputFile, OutputDirectory


class SkimsFile(RawOutputFile):
    """
    Represents a skims file used in activity-based models.
    Handles loading OMX files from local or remote paths.

    Attributes:
        (inherits attributes from RawOutputFile)
    """

    def __init__(self, inputDirectory: OutputDirectory):
        """
        Initializes a SkimsFile instance.

        Parameters:
            inputDirectory (OutputDirectory): The directory where the skims file is located.
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
        m.update(inputDirectory.directoryPath.encode())
        m.update(str(self._relativePathAttempt).encode())  # Convert list/str to bytes
        return m.hexdigest()
