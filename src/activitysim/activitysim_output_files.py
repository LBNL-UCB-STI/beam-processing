import hashlib
import os
import shutil
import urllib.request
from typing import Dict
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm

from src.input_base import RawOutputFile, InputDirectory
from src.constants import TMP_DIR


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
