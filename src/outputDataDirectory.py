import urllib.request
from typing import Tuple, Dict
from urllib.error import HTTPError

import urllib3

from src.input import (
    BeamRunInputDirectory,
    ActivitySimRunInputDirectory,
    PilatesRunInputDirectory,
    SfBayGeometry,
    Geometry,
)
from src.outputDataFrame import (
    PathTraversalEvents,
    PersonEntersVehicleEvents,
    ModeChoiceEvents,
    ModeVMT,
    LinkStatsFromPathTraversals,
    ProcessedPersonsFile,
    MandatoryLocationsByTaz,
    ProcessedHouseholdsFile,
    MandatoryLocationByTazByYear,
    ProcessedTripsFile,
    TripModeCount,
    ProcessedSkimsFile,
    TripModeCountByYear,
    TripModeCountByOrigin,
    TripPMT,
    TripPMTByOrigin,
    TripPMTByPrimaryPurpose,
    TripModeCountByPrimaryPurpose,
    ModeVMTByYear,
)


class OutputDataDirectory:
    """
    Represents an output data directory where results of postprocessing will be saved.

    Attributes:
        path (str): The path to the output data directory.
    """

    def __init__(self, path):
        self.path = path


class BeamOutputData:
    """
    Represents output data related to a Beam run.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        beamRunInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        pathTraversalEvents (src.outputDataFrame.PathTraversalEvents): Path traversal events data.
        personEntersVehicleEvents (src.outputDataFrame.PersonEntersVehicleEvents): Person enters vehicle events data.
        modeChoiceEvents (src.outputDataFrame.ModeChoiceEvents): Mode choice events data.
        modeVMT (src.outputDataFrame.ModeVMT): Mode vehicle miles traveled data.
        linkStatsFromPathTraversals (src.outputDataFrame.LinkStatsFromPathTraversals): Alternative linkstats
    """

    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        beamRunInputDirectory: BeamRunInputDirectory,
        collectEvents=False,
    ):
        """
        Initializes a BeamOutputData instance.

        Parameters:
            outputDataDirectory (OutputDataDirectory): The output data directory.
            beamRunInputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        """
        self.outputDataDirectory = outputDataDirectory
        self.beamRunInputDirectory = beamRunInputDirectory
        self.logFileRequest = urllib.request.Request(
            beamRunInputDirectory.append("beamLog.out")
        )
        self.logFileRequest.get_method = lambda: "HEAD"
        self.logFile = urllib.request.urlopen(self.logFileRequest)

        if collectEvents:
            self.beamRunInputDirectory.eventsFile.collectEvents(
                ["PathTraversal", "PersonEntersVehicle", "ModeChoice"]
            )

        self.pathTraversalEvents = PathTraversalEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.personEntersVehicleEvents = PersonEntersVehicleEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.modeChoiceEvents = ModeChoiceEvents(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.modeVMT = ModeVMT(self.outputDataDirectory, self.pathTraversalEvents)
        self.linkStatsFromPathTraversals = LinkStatsFromPathTraversals(
            self.outputDataDirectory, self.pathTraversalEvents
        )


class ActivitySimOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        activitySimRunInputDirectory: ActivitySimRunInputDirectory,
        skims: ProcessedSkimsFile,
        geometry=Geometry(),
    ):
        self.outputDataDirectory = outputDataDirectory
        self.activitySimRunInputDirectory = activitySimRunInputDirectory
        self.skims = skims
        self.geometry = geometry
        self.logFileRequest = urllib.request.Request(
            activitySimRunInputDirectory.append("final_land_use.csv.gz")
        )
        self.logFileRequest.get_method = lambda: "HEAD"
        self.logFile = urllib.request.urlopen(self.logFileRequest)

        self.persons = ProcessedPersonsFile(
            self.outputDataDirectory, self.activitySimRunInputDirectory
        )

        self.households = ProcessedHouseholdsFile(
            self.outputDataDirectory, self.activitySimRunInputDirectory
        )

        self.trips = ProcessedTripsFile(
            self.outputDataDirectory, self.activitySimRunInputDirectory
        )

        self.mandatoryLocationsByTaz = MandatoryLocationsByTaz(
            self.outputDataDirectory, self.persons
        )
        self.tripPMT = TripPMT(self.outputDataDirectory, self.trips, self.skims)
        self.tripPMTByOrigin = TripPMTByOrigin(
            self.outputDataDirectory, self.trips, self.skims
        )
        self.tripPMTByPrimaryPurpose = TripPMTByPrimaryPurpose(
            self.outputDataDirectory, self.trips, self.skims
        )
        self.tripModeCount = TripModeCount(self.outputDataDirectory, self.trips)
        self.tripModeCountByOrigin = TripModeCountByOrigin(
            self.outputDataDirectory, self.trips
        )
        self.tripModeCountByPrimaryPurpose = TripModeCountByPrimaryPurpose(
            self.outputDataDirectory, self.trips
        )


class PilatesOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        region="SFBay",
    ):
        self.outputDataDirectory = outputDataDirectory
        self.pilatesRunInputDirectory = pilatesRunInputDirectory
        self.asimRuns = dict[Tuple[int, int], ActivitySimOutputData]()
        self.beamRuns = dict[Tuple[int, int], BeamOutputData]()
        self.skims = ProcessedSkimsFile(
            self.outputDataDirectory, self.pilatesRunInputDirectory
        )
        if region == "SFBay":
            self.geometry = SfBayGeometry()
        else:
            self.geometry = Geometry()

        for (yr, it), directory in pilatesRunInputDirectory.asimRuns.items():
            try:
                self.asimRuns[(yr, it)] = ActivitySimOutputData(
                    outputDataDirectory, directory, self.skims, self.geometry
                )
            except HTTPError:
                print("Skipping ASim year {0} iteration {1}".format(yr, it))

        for (yr, it), directory in pilatesRunInputDirectory.beamRuns.items():
            try:
                self.beamRuns[(yr, it)] = BeamOutputData(outputDataDirectory, directory)
            except HTTPError:
                print("Skipping BEAM year {0} iteration {1}".format(yr, it))

        self.mandatoryLocationsByTazByYear = MandatoryLocationByTazByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )

        self.tripModeCountPerYear = TripModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.modeVMTPerYear = ModeVMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
