import os
from typing import Optional

from src.activitysim.activitysim_input_directory import ActivitySimRunInputDirectory
from src.geometry import Geometry
from src.input_base import gcs_blob_exists
from src.activitysim.activitysim_processed_data_frame import (
    ProcessedPersonsFile,
    ProcessedHouseholdsFile,
    ProcessedTripsFile,
    ProcessedToursFile,
    ProcessedSkimsFile,
    MandatoryLocationsByTaz,
    TripModeCount,
    TourModeCount,
    TripPMT,
    TripModeCountByOrigin,
    TripModeCountByPrimaryPurpose,
    TripPMTByOrigin,
    TripPMTByPrimaryPurpose,
)
from src.output_container import ModelOutputData, OutputDataDirectory


class ActivitySimOutputData(ModelOutputData):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        activitySimRunInputDirectory: ActivitySimRunInputDirectory,
        skims: Optional[ProcessedSkimsFile] = None,
        geometry: Optional[Geometry] = Geometry(),
    ):
        super().__init__(outputDataDirectory, activitySimRunInputDirectory)
        assert isinstance(self.inputDirectory, ActivitySimRunInputDirectory)
        self.skims = skims
        self.geometry = geometry
        lu_file_path = activitySimRunInputDirectory.append("final_land_use.csv.gz")
        if self.remoteResults and lu_file_path.startswith("gs://"):
            print(f"Checking status for GCS file: {lu_file_path}")
            if gcs_blob_exists(lu_file_path):
                self.logFileStatus = 200  # Simulate success status code
            else:
                self.logFileStatus = 404  # Simulate not found status code
        else:
            # Optionally check for local file existence
            local_lu_path = activitySimRunInputDirectory.append("final_land_use.csv.gz")
            self.logFileStatus = 200 if os.path.exists(local_lu_path) else 404

        # Initialize OutputDataFrame objects for ASIM outputs
        # These should return single DataFrames
        self.persons = ProcessedPersonsFile(
            self.outputDataDirectory, self.inputDirectory
        )

        self.households = ProcessedHouseholdsFile(
            self.outputDataDirectory, self.inputDirectory
        )

        self.trips = ProcessedTripsFile(self.outputDataDirectory, self.inputDirectory)
        self.tours = ProcessedToursFile(self.outputDataDirectory, self.inputDirectory)

        # Aggregate dataframes based on processed ASIM outputs
        self.mandatoryLocationsByTaz = MandatoryLocationsByTaz(
            self.outputDataDirectory, self.persons, self.geometry
        )
        # Note: TripPMT and related classes require skims which might not be available for all ASIM runs
        # if skims are only defined for the base year in PilatesRunInputDirectory.
        # Add checks or ensure skims is always available.
        if self.skims is not None:
            self.tripPMT = TripPMT(self.outputDataDirectory, self.trips, self.skims)
            self.tripPMTByOrigin = TripPMTByOrigin(
                self.outputDataDirectory, self.trips, self.skims
            )
            self.tripPMTByPrimaryPurpose = TripPMTByPrimaryPurpose(
                self.outputDataDirectory, self.trips, self.skims
            )
        else:
            print("No skims provided.")
        self.tripModeCount = TripModeCount(
            self.outputDataDirectory, self.trips, self.geometry
        )
        self.tourModeCount = TourModeCount(
            self.outputDataDirectory, self.tours, self.geometry
        )
        self.tripModeCountByOrigin = TripModeCountByOrigin(
            self.outputDataDirectory, self.trips, self.geometry
        )
        self.tripModeCountByPrimaryPurpose = TripModeCountByPrimaryPurpose(
            self.outputDataDirectory, self.trips
        )
