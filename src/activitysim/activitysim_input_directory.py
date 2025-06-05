from src.geometry import Geometry
from src.input_base import InputDirectory
from src.activitysim.activitysim_output_files import (
    TripUtilitiesFiles,
    PersonsFile,
    HouseholdsFile,
    TripsFile,
    ToursFile,
)


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
