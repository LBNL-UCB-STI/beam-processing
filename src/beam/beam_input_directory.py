from typing import Optional

from src.geometry import Geometry, SfBayGeometry, AustinGeometry, SeattleGeometry
from src.input_base import OutputDirectory
from src.beam.beam_output_files import (
    EventsFile,
    LinkStatsFile,
    NetworkFile,
    InputPlansFile,
    ReplanningEventReasonFile,
    ScoreStatsFile,
)


class BeamRunOutputDirectory(OutputDirectory):
    """
    Represents an input directory specific to a BEAM run.

    Attributes:
        eventsFile (src.beam.beam_output_files.EventsFile): The events file for the Beam run.
        inputPlansFile (src.beam.beam_output_files.InputPlansFile): The input plans file for the Beam run.
        linkStatsFile (src.beam.beam_output_files.LinkStatsFile): The link stats file for the Beam run.
        (inherits attributes from InputDirectory)
    """

    def __init__(
        self,
        baseFolderName: str,
        numberOfIterations: int = 0,
        geometry: Optional[Geometry] = None,
        region: Optional[str] = None,
        file_format: Optional[str] = "csv",
    ):
        """
        Initializes a BeamRunInputDirectory instance.

        Parameters:
            baseFolderName (str): The base folder name for the Beam run.
            numberOfIterations (int): The number of iterations for the Beam run.
        """
        super().__init__(baseFolderName, file_format)
        self.eventsFile = EventsFile(self, numberOfIterations, file_format)
        self.numberOfIterations = numberOfIterations
        self.inputPlansFile = InputPlansFile(self)
        self.replanningEventReasonFile = ReplanningEventReasonFile(self)
        self.scoreStatsFile = ScoreStatsFile(self)
        self.__linkStatsFile = {
            numberOfIterations: LinkStatsFile(self, numberOfIterations)
        }
        if (region is not None) & (geometry is None):
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
        else:
            self.geometry = geometry
        self.networkFile = NetworkFile(self, self.geometry)

    def linkStatsFile(self, numberOfIterations: Optional[int] = None):
        if numberOfIterations is None:
            numberOfIterations = self.numberOfIterations
        if numberOfIterations not in self.__linkStatsFile:
            self.__linkStatsFile[numberOfIterations] = LinkStatsFile(
                self, numberOfIterations
            )
        return self.__linkStatsFile[numberOfIterations]
