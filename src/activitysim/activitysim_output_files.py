import hashlib
import os
import shutil
import urllib.request
import logging
from typing import Dict
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm

from src.input_base import RawOutputFile, OutputDirectory
from src.constants import TMP_DIR

# Set up logging
logger = logging.getLogger(__name__)


class TripUtilitiesFiles(RawOutputFile):
    def __init__(self, inputDirectory: OutputDirectory):
        relativePath = "trip_mode_choice.zip"
        super().__init__(inputDirectory, relativePath, index_col="person_id")

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
            logger.info(f"Attempting to load trip utilities from {self.filePath}")
            out = dict()
            # Temporary folder for extracted zip contents
            folderName = os.path.join(TMP_DIR, self.__hash())
            zip_temp_path = None

            try:
                # Ensure the temporary directory exists
                os.makedirs(TMP_DIR, exist_ok=True)

                # Check if the extracted folder already exists (cached extraction)
                sentinel_file = os.path.join(folderName, "extraction_complete.sentinel")
                if not os.path.exists(folderName) or not os.path.exists(sentinel_file):
                    logger.info(f"Temporary extraction folder {folderName} not found or incomplete. Attempting download and extraction.")

                    # Clean up potentially incomplete folder before re-extracting
                    if os.path.exists(folderName):
                        logger.info(f"Cleaning up incomplete extraction folder {folderName}")
                        shutil.rmtree(folderName)
                    os.makedirs(folderName)

                    # Download the zip file to a temporary file first
                    zip_temp_path = os.path.join(TMP_DIR, f"{self.hash()}_utilities.zip")
                    logger.info(f"Downloading zip file from {self.filePath}")
                    urllib.request.urlretrieve(self.filePath, zip_temp_path)
                    logger.info(f"Downloaded zip file to {zip_temp_path}")

                    # Extract the zip file
                    logger.info(f"Extracting zip file to {folderName}")
                    with ZipFile(zip_temp_path, "r") as zfile:
                        # Extract only the utility files
                        utilities_files = [
                            f for f in zfile.filelist
                            if f.filename.endswith("utilities.csv")
                        ]

                        if not utilities_files:
                            logger.warning("No utilities.csv files found in zip archive")
                            return None

                        for ls in tqdm(utilities_files, desc="Extracting utilities"):
                            # Construct target path within the temporary folder
                            target_path = os.path.join(folderName, ls.filename)
                            # Ensure parent directory exists
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            # Extract the file
                            zfile.extract(ls.filename, folderName)

                    # Create a sentinel file to mark successful extraction
                    with open(sentinel_file, "w") as f:
                        f.write("Extraction complete")

                    logger.info("Extraction successful.")

                # Now read the extracted files from the temporary folder
                logger.info(f"Reading extracted utility files from {folderName}")
                extracted_utilities_dir = os.path.join(folderName, "trip_mode_choice")

                if not os.path.exists(extracted_utilities_dir):
                    logger.error(f"Expected subdirectory '{extracted_utilities_dir}' not found after extraction.")
                    return None

                files = os.listdir(extracted_utilities_dir)
                utilities_files = [f for f in files if f.endswith("utilities.csv")]

                if not utilities_files:
                    logger.warning("No utilities.csv files found in extracted directory")
                    return None

                for file in tqdm(utilities_files, desc="Reading utility CSVs"):
                    # file is like '0_trip_mode_choice_utilities.csv'
                    try:
                        groupName_str = file.split("_")[0]
                        groupName = int(groupName_str)
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Could not parse division ID from filename: {file}. Error: {e}. Skipping.")
                        continue

                    file_path_full = os.path.join(extracted_utilities_dir, file)
                    try:
                        df = pd.read_csv(file_path_full, index_col="trip_id")
                        if df.empty:
                            logger.warning(f"Empty DataFrame loaded from {file}")
                        else:
                            out[groupName] = df
                            logger.debug(f"Successfully loaded {df.shape[0]} rows from {file}")
                    except Exception as read_e:
                        logger.error(f"Error reading extracted file {file_path_full}: {read_e}. Skipping.")

                if out:
                    # Concatenate all the dataframes from the dictionary
                    self._file = pd.concat(out, names=["division", "trip_id"])
                    logger.info(f"Successfully concatenated {len(out)} utility files into DataFrame with {self._file.shape[0]} rows.")
                else:
                    logger.warning("No utility files successfully read.")
                    self._file = None

            except (FileNotFoundError, HTTPError, URLError) as e:
                logger.error(f"Failed to download zip file from {self.filePath}: {e}")
                # Clean up the incomplete extraction folder
                if os.path.exists(folderName):
                    shutil.rmtree(folderName)
                self._file = None
            except Exception as e:
                logger.error(f"An unexpected error occurred during zip processing: {e}")
                if os.path.exists(folderName):
                    shutil.rmtree(folderName)
                self._file = None
            finally:
                # Clean up the temporary zip file
                if zip_temp_path and os.path.exists(zip_temp_path):
                    try:
                        os.remove(zip_temp_path)
                        logger.debug(f"Cleaned up temporary zip file: {zip_temp_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary zip file {zip_temp_path}: {e}")

        return self._file


class PersonsFile(RawOutputFile):
    def __init__(self, inputDirectory: OutputDirectory):
        if hasattr(inputDirectory, 'file_format') and inputDirectory.file_format == "csv":
            relativePath = "persons.csv.gz"
        else:
            relativePath = "persons"
        super().__init__(inputDirectory, relativePath, index_col="person_id")

    def split(self, person_id_to_division) -> tuple[Dict[str, pd.DataFrame], Dict[int, str]]:
        """
        Split persons data by division.

        Args:
            person_id_to_division: Mapping from person_id to division

        Returns:
            Tuple of (division_to_persons, household_id_to_division)
        """
        try:
            persons_df = self.file()
            if persons_df is None or persons_df.empty:
                logger.warning("Persons file is empty or None")
                return {}, {}

            households = persons_df["household_id"].reset_index()
            households["division"] = households["person_id"].map(person_id_to_division)

            # Filter out rows where division mapping failed
            households = households.dropna(subset=["division"])

            household_id_to_division = households.set_index("person_id")["division"].to_dict()
            gb = persons_df.groupby(person_id_to_division)
            division_to_persons = {f: d for f, d in gb}

            logger.info(f"Split persons data into {len(division_to_persons)} divisions")
            return division_to_persons, household_id_to_division

        except Exception as e:
            logger.error(f"Error splitting persons file: {e}")
            return {}, {}


class HouseholdsFile(RawOutputFile):
    def __init__(self, inputDirectory: OutputDirectory):
        if hasattr(inputDirectory, 'file_format') and inputDirectory.file_format == "csv":
            relativePath = "households.csv.gz"
        else:
            relativePath = "households"
        super().__init__(inputDirectory, relativePath, index_col="household_id")

    def split(self, household_id_to_division) -> tuple[Dict[str, pd.DataFrame], Dict[int, str]]:
        """
        Split households data by division.

        Args:
            household_id_to_division: Mapping from household_id to division

        Returns:
            Tuple of (division_to_households, empty_dict)
        """
        try:
            households_df = self.file()
            if households_df is None or households_df.empty:
                logger.warning("Households file is empty or None")
                return {}, {}

            gb = households_df.groupby(household_id_to_division)
            division_to_households = {f: d for f, d in gb}

            logger.info(f"Split households data into {len(division_to_households)} divisions")
            return division_to_households, {}

        except Exception as e:
            logger.error(f"Error splitting households file: {e}")
            return {}, {}


class TripsFile(RawOutputFile):
    def __init__(self, inputDirectory: OutputDirectory):
        if hasattr(inputDirectory, 'file_format') and inputDirectory.file_format == "csv":
            relativePath = "final_trips.csv.gz"
        else:
            relativePath = "trips"
        super().__init__(
            inputDirectory,
            relativePath,
            index_col="trip_id",
            dtype={"household_id": int, "person_id": int, "tour_id": int},
        )

    def split(self, trip_id_to_division) -> tuple[Dict[str, pd.DataFrame], Dict[int, str], Dict[int, str]]:
        """
        Split trips data by division and consolidate overlapping person sets.

        Args:
            trip_id_to_division: Mapping from trip_id to division

        Returns:
            Tuple of (division_to_trips, person_id_to_division, trip_id_to_division_new)
        """
        try:
            trips_df = self.file()
            if trips_df is None or trips_df.empty:
                logger.warning("Trips file is empty or None")
                return {}, {}, {}

            persons = trips_df["person_id"].reset_index()
            persons["division"] = persons["trip_id"].map(trip_id_to_division)

            # Filter out rows where division mapping failed
            persons = persons.dropna(subset=["division"])

            divToPersons = persons.groupby("division").agg({"person_id": "unique"})

            if divToPersons.empty:
                logger.warning("No valid divisions found after mapping")
                return {}, {}, {}

            # Consolidate overlapping person sets
            tempMap = {}
            tempMap[divToPersons.iloc[0].name] = set(divToPersons.iloc[0].person_id)

            for div, ps in divToPersons.iterrows():
                setPersons = set(ps.person_id)
                unassigned = True
                for finalDiv, finalPersons in tempMap.items():
                    if not setPersons.isdisjoint(finalPersons):
                        finalPersons.update(setPersons)
                        unassigned = False
                        break
                if unassigned:
                    tempMap[div] = setPersons

            # Second pass to further consolidate
            outputMap = {}
            outputMap[divToPersons.iloc[0].name] = tempMap[divToPersons.iloc[0].name]

            for div, setPersons in tempMap.items():
                unassigned = True
                for finalDiv, finalPersons in outputMap.items():
                    if not setPersons.isdisjoint(finalPersons):
                        finalPersons.update(setPersons)
                        unassigned = False
                        break
                if unassigned:
                    outputMap[div] = setPersons

            person_id_to_division = (
                pd.concat([
                    pd.MultiIndex.from_product(
                        [[a], list(b)], names=["division", "person_id"]
                    ).to_frame(False)
                    for a, b in outputMap.items()
                ])
                .set_index("person_id")
                .to_dict()["division"]
            )

            trips_df["division_new"] = trips_df["person_id"].map(person_id_to_division)
            gb = trips_df.groupby("division_new")
            division_to_trips = {f: d for f, d in gb}
            trip_id_to_division_new = trips_df["division_new"].to_dict()

            logger.info(f"Split trips data into {len(division_to_trips)} consolidated divisions")
            return division_to_trips, person_id_to_division, trip_id_to_division_new

        except Exception as e:
            logger.error(f"Error splitting trips file: {e}")
            return {}, {}, {}


class ToursFile(RawOutputFile):
    def __init__(self, inputDirectory: OutputDirectory):
        if hasattr(inputDirectory, 'file_format') and inputDirectory.file_format == "csv":
            relativePath = "final_tours.csv.gz"
        else:
            relativePath = "tours"
        super().__init__(
            inputDirectory,
            relativePath,
            index_col="tour_id",
            dtype={"household_id": int, "person_id": int, "trip_id": int},
        )
