import os
from typing import Dict, Tuple
from urllib.error import HTTPError

import pandas as pd

from src.activitysim.activitysim_output_container import ActivitySimOutputData
from src.beam.beam_output_container import BeamOutputData
from src.geometry import SfBayGeometry, AustinGeometry, SeattleGeometry, Geometry
from src.input_directories import PilatesRunOutputDirectory
from src.activitysim.activitysim_multiyear_processed_data_frame import (
    TripPMTByYear,
    TripPMTByPrimaryPurposeByYear,
    TripPMTByCountyByYear,
    MandatoryLocationByTazByYear,
    TripModeCountByIteration,
    TourModeCountByIteration,
    TripModeCountByYear,
    TourModeCountByYear
)
from src.activitysim.activitysim_processed_data_frame import ProcessedSkimsFile
from src.beam.beam_multiyear_processed_data_frame import (
    ReplanningEventReasonByIteration,
    ScoreStatsByIteration,
    ModeVMTByYear,
    ModeEnergyByYear,
    ModePMTByYear,
    ModePMTByIteration,
    CongestionInfoByYear,
    CongestionInfoByIteration,
    PassengerMilesByVehicleAndModeByYear,
    PassengerMilesByVehicleAndModeByIteration,
    RealizedModeCountByIteration,
)
from src.output_container import ModelOutputData, OutputDataDirectory


class PilatesOutputData(ModelOutputData):
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        pilatesRunInputDirectory: PilatesRunOutputDirectory,
        region="SFBay",
        collectEvents: bool = False,
    ):
        super().__init__(outputDataDirectory, pilatesRunInputDirectory)
        self.outputDataDirectory = outputDataDirectory
        self.pilatesRunInputDirectory = pilatesRunInputDirectory
        self.asimRuns: Dict[Tuple[int, int], ActivitySimOutputData] = (
            {}
        )  # Specify dict type
        self.beamRuns: Dict[Tuple[int, int], BeamOutputData] = {}  # Specify dict type
        # The skims object should probably be initialized once per PilatesRunInputDirectory
        # and passed to each ActivitySimOutputData instance. This is already done.
        self.skims = ProcessedSkimsFile(
            self.outputDataDirectory, self.pilatesRunInputDirectory
        )
        if region == "SFBay":
            self.geometry = SfBayGeometry(
                otherFiles={
                    os.path.join(os.path.dirname(__file__), '..', "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv"): "zoneid"
                }
            )
        elif region == "Austin":
            self.geometry = AustinGeometry(otherFiles=dict())
        elif region == "Seattle":
            self.geometry = SeattleGeometry(otherFiles=dict())
        else:
            self.geometry = Geometry()

        # Initialize ActivitySimOutputData and BeamOutputData for each year/iteration
        for (yr, it), directory in pilatesRunInputDirectory.asimRuns.items():
            try:
                # Pass the shared skims object
                self.asimRuns[(yr, it)] = ActivitySimOutputData(
                    outputDataDirectory, directory, self.skims, self.geometry
                )
            except HTTPError:
                print("Skipping ASim year {0} iteration {1}".format(yr, it))
            except FileNotFoundError:  # Also catch local file not found
                print(
                    "Skipping ASim year {0} iteration {1} due to FileNotFoundError".format(
                        yr, it
                    )
                )

        for (yr, it), directory in pilatesRunInputDirectory.beamRuns.items():
            try:
                self.beamRuns[(yr, it)] = BeamOutputData(
                    outputDataDirectory, directory, collectEvents # Pass pilatesInputDict
                )
            except HTTPError:
                print("Skipping BEAM year {0} iteration {1}".format(yr, it))
            except FileNotFoundError:  # Also catch local file not found
                print(
                    "Skipping BEAM year {0} iteration {1} due to FileNotFoundError".format(
                        yr, it
                    )
                )

        # Initialize aggregated output objects that span years/iterations
        # These now use the __InfoByYear or __InfoByIteration base classes
        # and access data via accessors that point to the correct Beam/ASim run outputs.

        self.mandatoryLocationsByTazByYear = MandatoryLocationByTazByYear(
            self.outputDataDirectory,
            self.pilatesRunInputDirectory,
            pilatesInputDict=self.asimRuns,  # Pass the dictionary of ASIM runs as keyword arg
            geometry=self.geometry, # Pass geometry as keyword for clarity
        )

        self.tripPMTPerYear = TripPMTByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )
        self.tripPMTByPrimaryPurposePerYear = TripPMTByPrimaryPurposeByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )
        self.tripPMTByCountyPerYear = TripPMTByCountyByYear(
            self.outputDataDirectory,
            self.pilatesRunInputDirectory,
            pilatesInputDict=self.asimRuns, # Pass the dictionary of ASIM runs as keyword arg
            geometry=self.geometry, # Pass geometry as keyword for TAZBasedDataFrame
        )
        self.tripModeCountPerYear = TripModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )
        self.tourModeCountPerYear = TourModeCountByYear(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )
        self.tripModeCountPerIteration = TripModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )
        self.tourModeCountPerIteration = TourModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.asimRuns
        )

        self.replanningEventReasonPerIteration = ReplanningEventReasonByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.beamRuns
        )
        self.scoreStatsByIteration = ScoreStatsByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, pilatesInputDict=self.beamRuns
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
        self.congestionInfoByIteration = CongestionInfoByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )
        self.passengerMilesByVehicleAndModeByYear = (
            PassengerMilesByVehicleAndModeByYear(
                self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
            )
        )
        self.passengerMilesByVehicleAndModeByIteration = (
            PassengerMilesByVehicleAndModeByIteration(
                self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
            )
        )
        self.realizedModeCountyByIteration = RealizedModeCountByIteration(
            self.outputDataDirectory, self.pilatesRunInputDirectory, self.beamRuns
        )

    # Re-implement runInexus to call the new method on the specific BeamRunOutputData instance
    def runInexus(self, year, iter):
        """
        Runs the Inexus trip aggregation process for a specific year and iteration.

        Args:
            year (int): The simulation year.
            iter (int): The simulation iteration.

        Returns:
            pd.DataFrame: The aggregated person trips DataFrame for the specified run.
        """
        try:
            asimRun = self.asimRuns[(year, iter)]
            beamRun = self.beamRuns[(year, iter)]
        except KeyError:
            print(
                f"Error: Could not find ASIM or BEAM run for year {year}, iteration {iter}."
            )
            return pd.DataFrame()  # Return empty DataFrame on error

        # Call the new getAggregatedTrips method on the BeamRunOutputData instance
        return beamRun.getAggregatedTrips(asimRun.inputDirectory)
