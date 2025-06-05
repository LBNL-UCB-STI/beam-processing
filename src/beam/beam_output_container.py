import os
from multiprocessing import cpu_count

import pandas as pd
from joblib import Parallel, delayed

from src.activitysim.activitysim_input_directory import ActivitySimRunInputDirectory
from src.beam.beam_input_directory import BeamRunInputDirectory
from src.input_base import gcs_blob_exists
from src.output_container import ModelOutputData, OutputDataDirectory
from src.beam.beam_multiyear_processed_data_frame import NetworkVolumesByLinkByIteration
from src.beam.beam_processed_data_frame import (
    PathTraversalEvents,
    PersonTrips,
    PersonEntersVehicleEvents,
    ModeChoiceEvents,
    RealizedModeCount,
    ModeVMT,
    ModeVHT,
    PassengerMilesByVehicleAndMode,
    ReplanningEventReasons,
    ScoreStats,
    ModeEnergy,
    ModePMT,
    LinkStatsFromRawFile,
    LinkStatsFromPathTraversals,
    LabeledNetwork,
    NetworkVolumesByLink,
    LabeledLinkStatsFile,
    TAZTrafficVolumes,
)
from src.beam.beam_transformations import assignTripIdToEvents, mergeWithTripsAndAggregate


class BeamOutputData(ModelOutputData):
    """
    Represents output data related to a Beam run.

    Attributes:
        outputDataDirectory (src.output_container.OutputDataDirectory): The output data directory.
        inputDirectory (BeamRunInputDirectory): The input directory for the Beam run.
        pathTraversalEvents (src.beam.beam_processed_data_frame.PathTraversalEvents): Path traversal events data.
        personEntersVehicleEvents (src.beam.beam_processed_data_frame.PersonEntersVehicleEvents): Person enters vehicle events data.
        modeChoiceEvents (src.beam.beam_processed_data_frame.ModeChoiceEvents): Mode choice events data.
        modeVMT (src.beam.beam_processed_data_frame.ModeVMT): Mode vehicle miles traveled data.
        linkStatsFromPathTraversals (src.beam.beam_processed_data_frame.LinkStatsFromPathTraversals): Alternative linkstats
        personTrips (src.beam.beam_processed_data_frame.PersonTrips): Processed person trip events (dictionary of dataframes).
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
            outputDataDirectory (src.output_container.OutputDataDirectory): The output data directory.
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
            list(range(self.inputDirectory.numberOfIterations + 1)),
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
