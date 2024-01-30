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
        # self.logFile = urllib.request.urlopen(self.logFileRequest)
        self.geometry = beamRunInputDirectory.geometry

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

        self.personTrips = PersonTrips(
            self.outputDataDirectory, self.beamRunInputDirectory
        )

        self.modeVMT = ModeVMT(self.outputDataDirectory, self.pathTraversalEvents)
        self.modeEnergy = ModeEnergy(self.outputDataDirectory, self.pathTraversalEvents)
        self.linkStatsFromPathTraversals = LinkStatsFromPathTraversals(
            self.outputDataDirectory,
            self.pathTraversalEvents,
            self.beamRunInputDirectory.numberOfIterations,
        )
        self.labeledNetwork = LabeledNetwork(
            self.outputDataDirectory, self.beamRunInputDirectory
        )
        self.labeledLinkStatsFile = LabeledLinkStatsFile(
            self.outputDataDirectory,
            self.beamRunInputDirectory.linkStatsFile(),
            self.labeledNetwork,
            self.geometry,
        )
        self.tazTrafficVolumes = TAZTrafficVolumes(
            self.outputDataDirectory, self.labeledLinkStatsFile, self.geometry
        )
        self.networkVolumesByLink = NetworkVolumesByLink(
            self.outputDataDirectory,
            self.beamRunInputDirectory.linkStatsFile(self.beamRunInputDirectory.numberOfIterations),
            self.labeledNetwork,
        )
        self.networkVolumesByLinkByIteration = NetworkVolumesByLinkByIteration(
            self.outputDataDirectory,
            self.beamRunInputDirectory,
            self.labeledNetwork,
            list(range(self.beamRunInputDirectory.numberOfIterations)),
        )


class ActivitySimOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        activitySimRunInputDirectory: ActivitySimRunInputDirectory,
        skims: ProcessedSkimsFile,
        geometry: Optional[Geometry] = Geometry(),
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
        self.tripPMTByCountyPerYear = TripPMTByCountyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripModeCountPerYear = TripModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.tripModeCountByCountyPerYear = TripModeCountByCountyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.asimRuns
        )
        self.modeVMTPerYear = ModeVMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.modeEnergyPerYear = ModeEnergyByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.congestionInfoByYear = CongestionInfoByYear(
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
        ) = asimRun.activitySimRunInputDirectory.getSplitData()

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
        """      
        # Here's an example of how to group by county and road type
        look = self._runs["base"].beamRuns[(2010, -1)].tazTrafficVolumes
        look.process(
            dict(),
            ["county", "hour", "attributeOrigType"],
            {"VMT": "sum", "VHT": "sum"},
        )
        """

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
                self._popsByRegionType[
                    scenarioName
                ] = data.mandatoryLocationsByTazByYear.process(
                    normalize={"population": "area", "jobs": "area"},
                    aggregateBy=["areatype10", "year"],
                    mapping={"population": "sum", "jobs": "sum"},
                )
        return pd.concat(
            self._popsByRegionType,
            names=["scenario"] + self._popsByRegionType[scenarioName].index.names,
        )

    @property
    def populationByCountyAndRegionType(self):
        if len(self._popsByCountyAndRegionType) == 0:
            for scenarioName, data in self._runs.items():
                self._popsByCountyAndRegionType[
                    scenarioName
                ] = data.mandatoryLocationsByTazByYear.process(
                    normalize={"population": "area", "jobs": "area"},
                    aggregateBy=["county", "areatype10", "year"],
                    mapping={"population": "sum", "jobs": "sum"},
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
                self._popsByCounty[
                    scenarioName
                ] = data.mandatoryLocationsByTazByYear.process(
                    normalize={"population": "area", "jobs": "area"},
                    aggregateBy=["county", "year"],
                    mapping={"population": "sum", "jobs": "sum"},
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
            names=["scenario"] + data.tripModeCountPerYear.dataFrame.index.names,
        )

    @property
    def tripModeCountByCounty(self):
        if len(self._modeChoicesByCounty) == 0:
            for scenarioName, data in self._runs.items():
                self._modeChoicesByCounty[
                    scenarioName
                ] = data.tripModeCountByCountyPerYear.dataFrame
        return pd.concat(
            self._modeChoicesByCounty,
            names=["scenario"]
            + data.tripModeCountByCountyPerYear.dataFrame.index.names,
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
            names=["scenario"] + data.modeVMTPerYear.dataFrame.index.names,
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
            names=["scenario"] + data.modeEnergyPerYear.dataFrame.index.names,
        )
