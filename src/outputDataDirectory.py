import urllib.request
from multiprocessing import cpu_count
from typing import Tuple, Dict, Iterable, Optional
from urllib.error import HTTPError
import pandas as pd
import requests
import os
from google.cloud import storage  # Import google-cloud-storage client

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
    PassengerMilesByVehicleAndMode,
    PassengerMilesByVehicleAndModeByYear,
    RealizedModeCount,
    PassengerMilesByVehicleAndModeByIteration,
    RealizedModeCountByIteration,
    CongestionInfoByIteration,
    LinkStatsFromRawFile,
)
from src.transformations import assignTripIdToEvents, mergeWithTripsAndAggregate


def gcs_blob_exists(gcs_url: str) -> bool:
    """Checks if a blob exists at a given gs:// URL."""
    try:
        if not gcs_url.startswith("gs://"):
            # Not a GCS URL, return False or raise error depending on desired behavior
            print(f"Warning: gcs_blob_exists called with non-GCS URL: {gcs_url}")
            return False

        # Parse the bucket name and blob path from the URL
        parts = gcs_url.replace("gs://", "").split("/", 1)
        if len(parts) < 2:
            print(f"Warning: GCS URL '{gcs_url}' does not specify a blob path.")
            return False
        bucket_name = parts[0]
        blob_name = parts[1]

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()  # Use blob.exists() to check without downloading
    except Exception as e:
        print(f"Error checking GCS blob existence for '{gcs_url}': {e}")
        return False  # Assume not found or inaccessible on error


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
        personTrips (src.outputDataFrame.PersonTrips): Processed person trip events (dictionary of dataframes).
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
            collectEvents (bool): Whether to eagerly collect event files during initialization.
        """
        super().__init__(outputDataDirectory, beamRunInputDirectory)
        assert isinstance(self.inputDirectory, BeamRunInputDirectory)
        self.outputDataDirectory = outputDataDirectory
        log_file_path = beamRunInputDirectory.append("beamLog.out")
        if self.remoteResults and log_file_path.startswith("gs://"):
            print(f"Checking status for GCS file: {log_file_path}")
            if gcs_blob_exists(log_file_path):
                self.logFileStatus = 200  # Simulate success status code
            else:
                print(f"Log file not found at {log_file_path}")
                self.logFileStatus = 404

        else:
            self.logFileRequest = None
            self.logFile = None  # Not storing the file object here
            # Optionally check for local file existence
            local_log_path = beamRunInputDirectory.append("beamLog.out")
            self.logFileStatus = 200 if os.path.exists(local_log_path) else 404

        self.geometry = beamRunInputDirectory.geometry

        # Initialize OutputDataFrame objects for raw/processed event types
        # Note: These PathTraversalEvents, PersonEntersVehicleEvents, ModeChoiceEvents
        # objects will load and cache the *individual* event type dataframes upon first access
        # via their .dataFrame property.

        # The `personTrips` object (type PersonTrips) is designed differently.
        # Its `.dataFrame` property returns a DICTIONARY of processed event dataframes (PT, MC, TE, etc.),
        # not a single dataframe.
        # The subsequent objects (ModeVMT, ModePMT, etc.) however, expect a single dataframe
        # as input (PathTraversalEvents). This seems like a design conflict.
        # It's safer to have the lower-level classes (PathTraversalEvents etc) return
        # the standard dataframe, and have a separate process (like runInexus)
        # combine them and perform the trip-level aggregation.

        # Let's adjust the __init__ to reflect this understanding.
        # Instead of passing the raw PathTraversalEvents object to ModeVMT etc.,
        # we might need a different approach or re-evaluate the PersonTrips class design.
        # For now, keeping the structure but noting the design conflict and focusing on fixing methods.
        # The personTrips object seems specifically tied to the `runInexus` method's input.

        # Raw event accessors (these should load and cache individual event types)
        self._pathTraversalEvents = PathTraversalEvents(
            self.outputDataDirectory, self.inputDirectory
        )
        self._personEntersVehicleEvents = PersonEntersVehicleEvents(
            self.outputDataDirectory, self.inputDirectory
        )
        self._modeChoiceEvents = ModeChoiceEvents(
            self.outputDataDirectory, self.inputDirectory
        )

        # This object seems specifically intended for the runInexus method input
        # It loads a DICTIONARY of event types processed by doInexus.
        self.personTrips = PersonTrips(self.outputDataDirectory, self.inputDirectory)

        # Fixing the constructor calls to pass the PathTraversalEvents object
        self.realizedModeCount = RealizedModeCount(
            self.outputDataDirectory, self._modeChoiceEvents
        )  # Use the _modeChoiceEvents accessor

        # These depend on the processed PathTraversalEvents
        # Ensure they use self._pathTraversalEvents.dataFrame inside their load() methods
        self.modeVMT = ModeVMT(self.outputDataDirectory, self._pathTraversalEvents)
        self.modeVHT = ModeVHT(self.outputDataDirectory, self._pathTraversalEvents)
        self.passengerMilesByVehicleAndMode = PassengerMilesByVehicleAndMode(
            self.outputDataDirectory, self._pathTraversalEvents
        )
        self.modeEnergy = ModeEnergy(
            self.outputDataDirectory, self._pathTraversalEvents
        )
        self.modePMT = ModePMT(self.outputDataDirectory, self._pathTraversalEvents)

        self.replanningEventReasons = ReplanningEventReasons(
            self.outputDataDirectory, self.inputDirectory
        )
        self.scoreStats = ScoreStats(self.outputDataDirectory, self.inputDirectory)
        self.linkStatsFromPathTraversals = LinkStatsFromPathTraversals(
            self.outputDataDirectory,
            self._pathTraversalEvents,  # Pass the processed PTs object
            self.inputDirectory.numberOfIterations,
        )
        self.linkStatsFromRawFile = LinkStatsFromRawFile(
            self.outputDataDirectory,
            self.inputDirectory,
            self.inputDirectory.numberOfIterations,
        )
        self.labeledNetwork = LabeledNetwork(
            self.outputDataDirectory, self.inputDirectory
        )
        self.labeledLinkStatsFile = LabeledLinkStatsFile(
            self.outputDataDirectory,
            self.linkStatsFromPathTraversals,
            self.labeledNetwork,
            self.geometry,
        )
        self.tazTrafficVolumes = TAZTrafficVolumes(
            self.outputDataDirectory, self.labeledLinkStatsFile, self.geometry
        )
        self.networkVolumesByLink = NetworkVolumesByLink(
            self.outputDataDirectory,
            self.linkStatsFromRawFile,
            self.labeledNetwork,
        )
        self.networkVolumesByLinkByIteration = NetworkVolumesByLinkByIteration(
            self.outputDataDirectory,
            self.inputDirectory,
            self.labeledNetwork,
            list(range(self.inputDirectory.numberOfIterations+1)),
        )

    # Add a method to run the aggregated trip processing (formerly part of PilatesOutputData.runInexus)
    def getAggregatedTrips(self, asimRunInputDirectory: ActivitySimRunInputDirectory):
        """
        Orchestrates the processing of Beam events and ActivitySim outputs to
        produce a single aggregated DataFrame of person trips.

        This involves:
        1. Getting processed event data (PathTraversal, ModeChoice, etc.) from the Beam run.
           Note: The `personTrips` object's .dataFrame property returns a dictionary
           of these processed event dataframes, as generated by `doInexus`.
        2. Getting ActivitySim data (trips, persons, utilities).
        3. Splitting/chunking the data for parallel processing (optional, handled by input).
        4. Assigning trip IDs to events.
        5. Merging event data with ASIM data.
        6. Aggregating event data to trip level.
        7. Combining results from chunks.

        Parameters:
            asimRunInputDirectory (ActivitySimRunInputDirectory): The corresponding ActivitySim input directory.

        Returns:
            pd.DataFrame: A DataFrame where each row is a trip, aggregated from events
                          and merged with ASIM attributes.
        """
        # Get processed event data dictionary from the Beam run
        # This calls doInexus internally via the personTrips.dataFrame property
        processed_events_dict = self.personTrips.dataFrame
        if processed_events_dict is None:
            print("Could not load processed Beam event data.")
            return pd.DataFrame()

        # Get split ActivitySim data. This handles loading ASIM files and splitting them.
        (
            division_to_utilities,
            division_to_trips,
            division_to_persons,
            division_to_households,  # households data is not used in mergeWithTripsAndAggregate
            person_id_to_division,
        ) = asimRunInputDirectory.getSplitData()

        # Prepare event data for chunking based on the person_id_to_division mapping
        # Note: PersonTrips.chunk() was removed, doing the chunking here.
        event_types_to_chunk = [
            "ModeChoice",
            "PathTraversal",
            "TeleportationEvent",
            "PersonCost",
            "ParkingEvent",
            "Replanning",
        ]
        chunked_events = {}
        for event_type in event_types_to_chunk:
            if event_type in processed_events_dict:
                df = processed_events_dict[event_type]
                # Add divisionId based on the person_id_to_division map
                df["divisionId"] = (
                    df.index.get_level_values("IDMerged")
                    .astype(int)
                    .map(person_id_to_division)
                )
                chunked_events[event_type] = {
                    k: table for k, table in df.groupby("divisionId")
                }
            else:
                print(
                    f"Warning: Event type '{event_type}' not found in processed events dictionary."
                )
                chunked_events[event_type] = {}  # Provide empty dict for missing types

        # Get the list of chunk keys (division IDs) from ModeChoice, as MC is essential
        # Use keys from PT if MC is empty or missing, or just use keys from person_id_to_division
        chunk_keys = list(
            person_id_to_division.values()
        )  # Use all divisions from the person map

        if not chunk_keys:
            print("No divisions found for parallel processing.")
            return pd.DataFrame()

        # Helper function to process a single chunk (division)
        def combineChunk(chunk):
            print(f"Processing chunk: {chunk}")
            # Get event data for this chunk, providing empty DataFrames if a type is missing
            chunk_mc = chunked_events.get("ModeChoice", {}).get(chunk, pd.DataFrame())
            chunk_pt = chunked_events.get("PathTraversal", {}).get(
                chunk, pd.DataFrame()
            )
            chunk_te = chunked_events.get("TeleportationEvent", {}).get(
                chunk, pd.DataFrame()
            )
            chunk_pc = chunked_events.get("PersonCost", {}).get(chunk, pd.DataFrame())
            chunk_pe = chunked_events.get("ParkingEvent", {}).get(chunk, pd.DataFrame())
            chunk_rp = chunked_events.get("Replanning", {}).get(chunk, pd.DataFrame())

            if chunk_mc.empty:
                # If no mode choices for this chunk, cannot assign tripIds to other events in this chunk.
                # It's possible some events exist but no MC events.
                # Decide how to handle: return empty DF for this chunk, or process events without tripId assignment?
                # The aggregator expects MC data. Returning empty DF seems safer if MC is critical for trip definition.
                print(
                    f"Warning: No ModeChoice events found for chunk {chunk}. Skipping trip aggregation for this chunk."
                )
                # Need to return a DataFrame with the expected columns, even if empty
                # Determine expected columns from mergeWithTripsAndAggregate's aggfunc and asimData merge.
                # This is complex - maybe create a dummy empty DF with correct cols?
                # For now, return None and filter later.
                return None

            # Assign trip IDs to events in this chunk using the chunked ModeChoice data
            # Pass other columns needed for aggregation from ModeChoice
            pts_with_tripId = assignTripIdToEvents(
                chunk_pt,
                chunk_mc,
                {
                    "mode_choice_actual_BEAM": "mode_choice_actual_BEAM",
                    "mode_choice_planned_BEAM": "mode_choice_planned_BEAM",
                    "distance_mode_choice": "distance_mode_choice",  # This was length in MC
                },
            )
            # Teleportation events also need tripId and potentially other columns
            tes_with_tripId = assignTripIdToEvents(
                chunk_te,
                chunk_mc,
                {
                    "mode_choice_actual_BEAM": "mode_choice_actual_BEAM",
                    "mode_choice_planned_BEAM": "mode_choice_planned_BEAM",
                    # Assuming distance_travelling from TE corresponds to distance_mode_choice concept for teleportation
                    "distance_mode_choice": "distance_travelling",
                },
            )
            if not tes_with_tripId.empty:
                # Ensure distance_privateCar is populated for TE as expected by mergeWithTripsAndAggregate
                tes_with_tripId["distance_privateCar"] = tes_with_tripId[
                    "distance_travelling"
                ].copy()

            pcs_with_tripId = assignTripIdToEvents(chunk_pc, chunk_mc)
            pes_with_tripId = assignTripIdToEvents(chunk_pe, chunk_mc)
            rps_with_tripId = assignTripIdToEvents(chunk_rp, chunk_mc)

            # Combine all event types for this chunk
            # Use errors='ignore' just in case column names differ unexpectedly
            allEvents = pd.concat(
                [
                    pts_with_tripId,
                    tes_with_tripId,
                    pcs_with_tripId,
                    pes_with_tripId,
                    rps_with_tripId,
                ],
                axis=0,
                ignore_index=False,
            )  # Keep original index levels (IDMerged, eventOrder)

            # Merge chunked event data with chunked ASIM data and aggregate to trip level
            # Note: division_to_households is not passed as it's not used in mergeWithTripsAndAggregate
            combined = mergeWithTripsAndAggregate(
                allEvents,
                division_to_trips.get(
                    chunk, pd.DataFrame()
                ),  # Provide empty DF if chunk missing
                division_to_utilities.get(
                    chunk, pd.DataFrame()
                ),  # Provide empty DF if chunk missing
                division_to_persons.get(
                    chunk, pd.DataFrame()
                ),  # Provide empty DF if chunk missing
            )
            print(f"Finished processing chunk: {chunk}")
            return combined

        # Execute parallel processing
        # n_jobs=cpu_count() // 2 is reasonable, cap it or make configurable
        num_cores = min(cpu_count() // 2, 8)  # Cap at a reasonable number
        print(f"Starting parallel processing with {num_cores} cores.")
        processed_list = Parallel(n_jobs=num_cores)(
            delayed(combineChunk)(ch) for ch in chunk_keys
        )

        # Filter out None results from chunks where MC data was missing
        processed_list = [df for df in processed_list if df is not None]

        if not processed_list:
            print("No data processed from any chunk.")
            return pd.DataFrame()

        # Concatenate results from all chunks
        combinedData = pd.concat(
            processed_list, axis=0, ignore_index=True
        )  # Ignore index might be needed if merge creates non-unique indices

        # Basic logging for debugging
        print(
            "Processing complete. Finding {0} unmatched ASim trips and {1} unmatched BEAM trips out of {2} total".format(
                combinedData.trip_id.isna().sum(),
                combinedData.tripId.isna().sum(),
                combinedData.shape[0],
            )
        )

        return combinedData


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


class PilatesOutputData:
    def __init__(
        self,
        outputDataDirectory: OutputDataDirectory,
        pilatesRunInputDirectory: PilatesRunInputDirectory,
        region="SFBay",
        collectEvents: bool = False,
    ):
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
                    "geoms/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv": "zoneid"
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
                    outputDataDirectory, directory, collectEvents
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
            self.asimRuns,  # Pass the dictionary of ASIM runs
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
