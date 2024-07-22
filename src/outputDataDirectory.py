import urllib.request
from multiprocessing import cpu_count
from typing import Tuple, Dict, Iterable, Optional
from urllib.error import HTTPError
import pandas as pd
import scipy as sp

import urllib3
from joblib import Parallel, delayed

from src.input import (
    BeamRunInputDirectory,
    ActivitySimRunInputDirectory,
    PilatesRunInputDirectory,
    SfBayGeometry,
    Geometry,
    AustinGeometry,
    SeattleGeometry,
    InputDirectory,
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
    ModeEnergy,
    TripModeCountByCountyByYear,
    ModeEnergyByYear,
    TripPMTByYear,
    TripPMTByCountyByYear,
    LabeledLinkStatsFile,
    LabeledNetwork,
    TAZTrafficVolumes,
    PersonTrips,
    CongestionInfoByYear,
    NetworkVolumesByLink,
    NetworkVolumesByLinkByIteration,
    TripsByYear,
    TripPMTByPrimaryPurposeByYear,
    ModePMT,
    ModePMTByYear,
    TripModeCountByIteration,
    ModePMTByIteration,
    ReplanningEventReasons,
    ReplanningEventReasonByIteration,
    ScoreStats,
    ScoreStatsByIteration,
    TourModeCountByIteration,
    TourModeCountByYear,
    TourModeCount,
    ProcessedToursFile,
    ModeVHT,
    PassengerMilesByVehicleAndMode, PassengerMilesByVehicleAndModeByYear, RealizedModeCount,
    PassengerMilesByVehicleAndModeByIteration, RealizedModeCountByIteration,
)
from src.transformations import assignTripIdToEvents, mergeWithTripsAndAggregate


class OutputDataDirectory:
    """
    Represents an output data directory where results of postprocessing will be saved.

    Attributes:
        path (str): The path to the output data directory.
    """

    def __init__(self, path):
        self.path = path


class ModelOutputData:
    def __init__(
        self, outputDataDirectory: OutputDataDirectory, inputDirectory: InputDirectory
    ):
        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        self.remoteResults = inputDirectory.isLink


class BeamOutputData(ModelOutputData):
    """
    Represents output data related to a Beam run.

    Attributes:
        outputDataDirectory (OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
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
        super().__init__(outputDataDirectory, beamRunInputDirectory)
        assert isinstance(self.inputDirectory, BeamRunInputDirectory)
        self.outputDataDirectory = outputDataDirectory
        if self.remoteResults:
            self.logFileRequest = urllib.request.Request(
                beamRunInputDirectory.append("beamLog.out")
            )
            self.logFileRequest.get_method = lambda: "HEAD"
            self.logFile = urllib.request.urlopen(self.logFileRequest)
        else:
            self.logFileRequest = None
            self.logFile = None
        self.geometry = beamRunInputDirectory.geometry

        if collectEvents:
            self.inputDirectory.eventsFile.collectEvents(
                ["PathTraversal", "PersonEntersVehicle", "ModeChoice"]
            )

        self.pathTraversalEvents = PathTraversalEvents(
            self.outputDataDirectory, self.inputDirectory
        )
        self.personEntersVehicleEvents = PersonEntersVehicleEvents(
            self.outputDataDirectory, self.inputDirectory
        )
        self.modeChoiceEvents = ModeChoiceEvents(
            self.outputDataDirectory, self.inputDirectory
        )

        if collectEvents:
            self.pathTraversalEvents.load()
            self.personEntersVehicleEvents.load()
            self.modeChoiceEvents.load()
            self.inputDirectory.eventsFile.clearEvents()

        self.personTrips = PersonTrips(
            self.outputDataDirectory, self.inputDirectory
        )
        self.realizedModeCount = RealizedModeCount(self.outputDataDirectory, self.modeChoiceEvents)

        self.modeVMT = ModeVMT(self.outputDataDirectory, self.pathTraversalEvents)
        self.modeVHT = ModeVHT(self.outputDataDirectory, self.pathTraversalEvents)
        self.passengerMilesByVehicleAndMode = PassengerMilesByVehicleAndMode(
            self.outputDataDirectory, self.pathTraversalEvents
        )
        self.modeEnergy = ModeEnergy(self.outputDataDirectory, self.pathTraversalEvents)
        self.modePMT = ModePMT(self.outputDataDirectory, self.pathTraversalEvents)
        self.replanningEventReasons = ReplanningEventReasons(
            self.outputDataDirectory, self.inputDirectory
        )
        self.scoreStats = ScoreStats(
            self.outputDataDirectory, self.inputDirectory
        )
        self.linkStatsFromPathTraversals = LinkStatsFromPathTraversals(
            self.outputDataDirectory,
            self.pathTraversalEvents,
            self.inputDirectory.numberOfIterations,
        )
        self.labeledNetwork = LabeledNetwork(
            self.outputDataDirectory, self.inputDirectory
        )
        self.labeledLinkStatsFile = LabeledLinkStatsFile(
            self.outputDataDirectory,
            self.inputDirectory.linkStatsFile(),
            self.labeledNetwork,
            self.geometry,
        )
        self.tazTrafficVolumes = TAZTrafficVolumes(
            self.outputDataDirectory, self.labeledLinkStatsFile, self.geometry
        )
        self.networkVolumesByLink = NetworkVolumesByLink(
            self.outputDataDirectory,
            self.inputDirectory.linkStatsFile(
                self.inputDirectory.numberOfIterations
            ),
            self.labeledNetwork,
        )
        self.networkVolumesByLinkByIteration = NetworkVolumesByLinkByIteration(
            self.outputDataDirectory,
            self.inputDirectory,
            self.labeledNetwork,
            list(range(self.inputDirectory.numberOfIterations)),
        )

    def collectAllEvents(self):
        self.inputDirectory.eventsFile.collectEvents(
            ["PathTraversal", "PersonEntersVehicle", "ModeChoice"]
        )
        self.pathTraversalEvents.dataFrame
        self.personEntersVehicleEvents.dataFrame
        self.modeChoiceEvents.dataFrame
        self.inputDirectory.eventsFile.clearEvents()


class ActivitySimOutputData(ModelOutputData):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        activitySimRunInputDirectory: ActivitySimRunInputDirectory,
        skims: ProcessedSkimsFile,
        geometry: Optional[Geometry] = Geometry(),
    ):
        super().__init__(outputDataDirectory, activitySimRunInputDirectory)
        assert isinstance(self.inputDirectory, ActivitySimRunInputDirectory)
        self.skims = skims
        self.geometry = geometry
        if self.remoteResults:
            self.logFileRequest = urllib.request.Request(
                activitySimRunInputDirectory.append("final_land_use.csv.gz")
            )
            self.logFileRequest.get_method = lambda: "HEAD"
            self.logFile = urllib.request.urlopen(self.logFileRequest)
        else:
            self.logFileRequest = None
            self.logFile = None

        self.persons = ProcessedPersonsFile(
            self.outputDataDirectory, self.inputDirectory
        )

        self.households = ProcessedHouseholdsFile(
            self.outputDataDirectory, self.inputDirectory
        )

        self.trips = ProcessedTripsFile(
            self.outputDataDirectory, self.inputDirectory
        )
        self.tours = ProcessedToursFile(
            self.outputDataDirectory, self.inputDirectory
        )

        self.mandatoryLocationsByTaz = MandatoryLocationsByTaz(
            self.outputDataDirectory, self.persons, self.geometry
        )
        self.tripPMT = TripPMT(self.outputDataDirectory, self.trips, self.skims)
        self.tripPMTByOrigin = TripPMTByOrigin(
            self.outputDataDirectory, self.trips, self.skims
        )
        self.tripPMTByPrimaryPurpose = TripPMTByPrimaryPurpose(
            self.outputDataDirectory, self.trips, self.skims
        )
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
            self.outputDataDirectory,
            self.pilatesRunInputDirectory,
            self.asimRuns,
            self.geometry,
        )

        self.tripPMTPerYear = TripPMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripPMTByPrimaryPurposePerYear = TripPMTByPrimaryPurposeByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripPMTByCountyPerYear = TripPMTByCountyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripModeCountPerYear = TripModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tourModeCountPerYear = TourModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripModeCountPerIteration = TripModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tourModeCountPerIteration = TourModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripModeCountByCountyPerYear = TripModeCountByCountyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.replanningEventReasonPerIteration = ReplanningEventReasonByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.scoreStatsByIteration = ScoreStatsByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.modeVMTPerYear = ModeVMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.modeEnergyPerYear = ModeEnergyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.modePMTPerYear = ModePMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.modePMTPerIteration = ModePMTByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.congestionInfoByYear = CongestionInfoByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.passengerMilesByVehicleAndModeByYear = PassengerMilesByVehicleAndModeByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.passengerMilesByVehicleAndModeByIteration = PassengerMilesByVehicleAndModeByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.realizedModeCountyByIteration = RealizedModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )

    def runInexus(self, year, iter):
        asimRun = self.asimRuns[(year, iter)]
        beamRun = self.beamRuns[(year, iter)]
        (
            division_to_utilities,
            division_to_trips,
            division_to_persons,
            division_to_households,
            person_id_to_division,
        ) = asimRun.inputDirectory.getSplitData()

        combinedData = beamRun.personTrips.chunk(person_id_to_division)
        mc = combinedData["ModeChoice"]
        pt = combinedData["PathTraversal"]
        te = combinedData["TeleportationEvent"]
        pc = combinedData["PersonCost"]
        pe = combinedData["ParkingEvent"]
        rp = combinedData["Replanning"]

        def combineChunk(chunk):
            pts = assignTripIdToEvents(
                pt[chunk],
                mc[chunk],
                {
                    "mode_choice_actual_BEAM": "mode_choice_actual_BEAM",
                    "mode_choice_planned_BEAM": "mode_choice_planned_BEAM",
                    "distance_mode_choice": "distance_mode_choice",
                },
            )
            tes = assignTripIdToEvents(
                te[chunk], mc[chunk], {"distance_mode_choice": "distance_travelling"}
            )
            tes["distance_privateCar"] = tes["distance_travelling"].copy()
            tes["distance_mode_choice"] = tes["distance_travelling"].copy()
            pcs = assignTripIdToEvents(pc[chunk], mc[chunk])
            pes = assignTripIdToEvents(pe[chunk], mc[chunk])
            rps = assignTripIdToEvents(rp[chunk], mc[chunk])
            allEvents = pd.concat([pts, tes, pcs, pes, rps], axis=0)
            combined = mergeWithTripsAndAggregate(
                allEvents,
                division_to_trips[chunk],
                division_to_utilities[chunk],
                division_to_persons[chunk],
            )
            return combined

        test = False
        if test:
            out = combineChunk(list(person_id_to_division.values())[0])
            print("Success!")

        processed_list = Parallel(n_jobs=cpu_count() // 2)(
            delayed(combineChunk)(ch) for ch in mc.keys()
        )

        combinedData = pd.concat(processed_list, axis=0)

        print(
            "Finding {0} unmatched ASim trips and {1} unmatched BEAM trips out of {2} total".format(
                combinedData.trip_id.isna().sum(),
                combinedData.tripId.isna().sum(),
                combinedData.shape[0],
            )
        )

        return combinedData


class PilatesSettings:
    def __init__(
        self,
        scenarioName: str,
        path: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations: int,
    ):
        self.scenarioName = scenarioName
        self.path = path
        self.years = years
        self.asimLiteIteratsions = asimLiteIterations
        self.beamIterations = beamIterations


class PilatesAnalysis:
    def __init__(self, allPilatesSettings: Iterable[PilatesSettings]):
        self.allPilatesSettings = allPilatesSettings
        self._runs = dict()
        for ps in self.allPilatesSettings:
            directory = PilatesRunInputDirectory(
                ps.path, ps.years, ps.asimLiteIteratsions, ps.beamIterations
            )
            self._runs[ps.scenarioName] = PilatesOutputData(
                OutputDataDirectory("output/{0}".format(ps.scenarioName)), directory
            )
        self._pops = dict()
        self._popsByCounty = dict()
        self._popsByRegionType = dict()
        self._popsByCountyAndRegionType = dict()
        self._modechoices = dict()
        self._modeChoicesByCounty = dict()
        self._pmtByCounty = dict()
        self._modeChoiceByPurpose = dict()
        self._pmtByPurpose = dict()
        self._modeVMT = dict()
        self._modeEnergy = dict()
        self._modePMT = dict()
        self._personTrips = dict()
        self.inexus = dict()
        """      
        # Here's an example of how to group by county and road type
        look = self._runs["base"].beamRuns[(2010, -1)].tazTrafficVolumes
        look.process(
            dict(),
            ["county", "hour", "attributeOrigType"],
            {"VMT": "sum", "VHT": "sum"},
        )
        """

    def runInexus(self):
        for ps in self.allPilatesSettings:
            self.inexus[ps.scenarioName] = self._runs[ps.scenarioName].runInexus(
                ps.years[-1], ps.asimLiteIteratsions
            )

    # @property
    # def personTrips(self):
    #     if len(self._pops) == 0:
    #         for scenarioName, data in self._runs.items():
    #             self._personTrips[scenarioName] = data.tripsByYear.dataFrame
    #     return pd.concat(
    #         self._personTrips, names=["scenario"] + self._pops[scenarioName].index.names
    #     )

    @property
    def populationByTaz(self):
        if len(self._pops) == 0:
            for scenarioName, data in self._runs.items():
                self._pops[scenarioName] = data.mandatoryLocationsByTazByYear.process(
                    normalize={"population": "area", "jobs": "area"}
                )
        return pd.concat(
            self._pops, names=["scenario"] + self._pops[scenarioName].index.names
        )

    @property
    def populationByRegionType(self):
        if len(self._popsByRegionType) == 0:
            for scenarioName, data in self._runs.items():
                self._popsByRegionType[scenarioName] = (
                    data.mandatoryLocationsByTazByYear.process(
                        normalize={"population": "area", "jobs": "area"},
                        aggregateBy=["areatype10", "year"],
                        mapping={"population": "sum", "jobs": "sum"},
                    )
                )
        return pd.concat(
            self._popsByRegionType,
            names=["scenario"] + self._popsByRegionType[scenarioName].index.names,
        )

    @property
    def populationByCountyAndRegionType(self):
        if len(self._popsByCountyAndRegionType) == 0:
            for scenarioName, data in self._runs.items():
                self._popsByCountyAndRegionType[scenarioName] = (
                    data.mandatoryLocationsByTazByYear.process(
                        normalize={"population": "area", "jobs": "area"},
                        aggregateBy=["county", "areatype10", "year"],
                        mapping={"population": "sum", "jobs": "sum"},
                    )
                )
        return pd.concat(
            self._popsByCountyAndRegionType,
            names=["scenario"]
            + self._popsByCountyAndRegionType[scenarioName].index.names,
        )

    @property
    def populationByCounty(self):
        if len(self._popsByCounty) == 0:
            for scenarioName, data in self._runs.items():
                self._popsByCounty[scenarioName] = (
                    data.mandatoryLocationsByTazByYear.process(
                        normalize={"population": "area", "jobs": "area"},
                        aggregateBy=["county", "year"],
                        mapping={"population": "sum", "jobs": "sum"},
                    )
                )
        return pd.concat(
            self._popsByCounty,
            names=["scenario"] + self._popsByCounty[scenarioName].index.names,
        )

    @property
    def tripModeCount(self):
        if len(self._modechoices) == 0:
            for scenarioName, data in self._runs.items():
                self._modechoices[scenarioName] = data.tripModeCountPerYear.dataFrame
        return pd.concat(
            self._modechoices,
            names=["scenario"]
            + list(self._runs.values())[0].tripModeCountPerYear.dataFrame.index.names,
        )

    @property
    def tourModeCount(self):
        if len(self._modechoices) == 0:
            for scenarioName, data in self._runs.items():
                self._modechoices[scenarioName] = data.tourModeCountPerYear.dataFrame
        return pd.concat(
            self._modechoices,
            names=["scenario"]
            + list(self._runs.values())[0].tourModeCountPerYear.dataFrame.index.names,
        )

    @property
    def pmtByPurpose(self):
        if len(self._pmtByPurpose) == 0:
            for scenarioName, data in self._runs.items():
                self._pmtByPurpose[scenarioName] = (
                    data.tripPMTByPrimaryPurposePerYear.dataFrame
                )
        return pd.concat(
            self._pmtByPurpose,
            names=["scenario"]
            + list(self._runs.values())[
                0
            ].tripPMTByPrimaryPurposePerYear.dataFrame.index.names,
        )

    @property
    def tripModeCountByCounty(self):
        if len(self._modeChoicesByCounty) == 0:
            for scenarioName, data in self._runs.items():
                self._modeChoicesByCounty[scenarioName] = (
                    data.tripModeCountByCountyPerYear.dataFrame
                )
        return pd.concat(
            self._modeChoicesByCounty,
            names=["scenario"]
            + list(self._runs.values())[
                0
            ].tripModeCountByCountyPerYear.dataFrame.index.names,
        )

    @property
    def vmtByMode(self):
        if len(self._modeVMT) == 0:
            for scenarioName, data in self._runs.items():
                try:
                    self._modeVMT[scenarioName] = data.modeVMTPerYear.dataFrame
                except HTTPError:
                    continue
        return pd.concat(
            {key: val for key, val in self._modeVMT.items() if len(val) > 0},
            names=["scenario"]
            + list(self._runs.values())[0].modeVMTPerYear.dataFrame.index.names,
        )

    @property
    def energyByMode(self):
        if len(self._modeEnergy) == 0:
            for scenarioName, data in self._runs.items():
                try:
                    self._modeEnergy[scenarioName] = data.modeEnergyPerYear.dataFrame
                except HTTPError:
                    continue
        return pd.concat(
            {key: val for key, val in self._modeEnergy.items() if len(val) > 0},
            names=["scenario"]
            + list(self._runs.values())[0].modeEnergyPerYear.dataFrame.index.names,
        )

    @property
    def pmtByMode(self):
        if len(self._modePMT) == 0:
            for scenarioName, data in self._runs.items():
                try:
                    self._modePMT[scenarioName] = data.modePMTPerYear.dataFrame
                except HTTPError:
                    continue
        return pd.concat(
            {key: val for key, val in self._modePMT.items() if len(val) > 0},
            names=["scenario"]
            + list(self._runs.values())[0].modePMTPerYear.dataFrame.index.names,
        )
