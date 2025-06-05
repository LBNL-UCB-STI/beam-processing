from typing import Iterable, Optional
from urllib.error import HTTPError
import pandas as pd

from src.input_directories import (
    PilatesRunInputDirectory,
)
from src.output_container import OutputDataDirectory
from src.pilates_output_container import PilatesOutputData


class PilatesSettings:
    def __init__(
        self,
        scenarioName: str,
        path: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations: int,
        region: Optional[str] = "Sfbay",
    ):
        self.scenarioName = scenarioName
        self.path = path
        self.years = years
        self.asimLiteIteratsions = asimLiteIterations
        self.beamIterations = beamIterations
        self.region = region


class PilatesAnalysis:
    def __init__(self, allPilatesSettings: Iterable[PilatesSettings]):
        self.allPilatesSettings = allPilatesSettings
        self._runs = dict()
        for ps in self.allPilatesSettings:
            directory = PilatesRunInputDirectory(
                ps.path,
                ps.years,
                ps.asimLiteIteratsions,
                ps.beamIterations,
                region=ps.region,
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
