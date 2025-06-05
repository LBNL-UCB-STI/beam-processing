from typing import Iterable

# Use requests for more flexible HTTP calls (e.g., HEAD)

import numpy as np

from src.activitysim.activitysim_input_directory import ActivitySimRunOutputDirectory
from src.beam.beam_input_directory import BeamRunOutputDirectory
from src.geometry import Geometry, SfBayGeometry, AustinGeometry, SeattleGeometry
from src.input_base import OutputDirectory
from src.skims_file import SkimsFile


class PilatesRunOutputDirectory(OutputDirectory):
    def __init__(
        self,
        baseFolderName: str,
        years: Iterable[int],
        asimLiteIterations: int,
        beamIterations=0,
        region="SFBay",
        file_format="csv",
        collectEvents=False,
    ):
        super().__init__(baseFolderName, file_format)
        self.asimRuns = dict()
        self.beamRuns = dict()
        self.file_format = file_format
        try:
            self.skims = SkimsFile(self)
        except Exception as e:
            print("Skipping skims")
        if region == "SFBay":
            self.geometry = SfBayGeometry(
                otherFiles={
                    "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
                }
            )
        elif region == "Seattle":
            self.geometry = SeattleGeometry(otherFiles=dict())
        elif region == "Austin":
            self.geometry = AustinGeometry(otherFiles=dict())
        else:
            self.geometry = Geometry()
        for year in years:
            for asimLiteIteration in [-1, *np.arange(asimLiteIterations) + 1]:
                relPath = ["activitysim"]
                if not self.isLink:
                    relPath.append("output")
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                print("Loading year {0} it {1}".format(year, asimLiteIteration))
                self.asimRuns[(year, asimLiteIteration)] = ActivitySimRunOutputDirectory(
                    self.append(relPath), self.geometry, file_format
                )
                relPath = ["beam"]
                if not self.isLink:
                    relPath.append("beam_output")
                    relPath.append(region.lower())
                relPath.append("year-{0}-iteration-{1}".format(year, asimLiteIteration))
                self.beamRuns[(year, asimLiteIteration)] = BeamRunOutputDirectory(
                    self.append(relPath),
                    beamIterations,
                    self.geometry,
                    region,
                    file_format,
                )
