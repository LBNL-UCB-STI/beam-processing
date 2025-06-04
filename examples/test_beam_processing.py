import os
import sys
import traceback  # Import traceback for detailed error logging

from src.input import SeattleGeometry
from src.outputDataFrame import ProcessedSkimsFile

# Add the parent directory of src to the Python path
# This assumes the script is run from the project root directory (beam-processing/)
# If running from a different directory, adjust the path accordingly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import modules from src
try:
    from src import input, outputDataDirectory
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Please ensure the script is run from the beam-processing directory")
    print(f"or that the parent directory '{parent_dir}' is in your Python path.")
    sys.exit(1)


# --- Configuration ---
# Google Cloud Storage path for the Pilates run
# !! NOTE: Ensure you have authenticated to Google Cloud Storage (e.g., `gcloud auth application-default login`)
#    or have the GOOGLE_APPLICATION_CREDENTIALS environment variable set.
BEAM_OUTPUT_PATH = "output/beam/seattle/year-2018-iteration--1"


# Region for geometry loading
REGION = "Seattle"

# Local directory to save processed outputs and cache
LOCAL_OUTPUT_BASE_PATH = "tmp-test"

# --- Script Execution ---

# Set GCLOUD_PROJECT environment variable for GCS access
# Replace with your Google Cloud project ID if necessary
os.environ["GCLOUD_PROJECT"] = "1010663794916"
print(f"Set GCLOUD_PROJECT to {os.environ['GCLOUD_PROJECT']}")
os.chdir("../")

# Ensure the local output directory exists
local_scenario_output_path = os.path.join(
    LOCAL_OUTPUT_BASE_PATH, "seattle_asim_test_run"
)
os.makedirs(local_scenario_output_path, exist_ok=True)
print(f"Local output directory: {local_scenario_output_path}")


print("\n--- Initializing Pilates Input and Output Data ---")
try:
    # Create the Pilates Input Directory object
    beamInputDirectory = input.BeamRunInputDirectory(
        BEAM_OUTPUT_PATH,
        numberOfIterations=0,
        geometry=SeattleGeometry(),
        region=REGION,
        file_format="parquet",
    )
    print("BeamRunInputDirectory initialized.")

    # Create the OutputDataDirectory object for saving processed outputs
    outputDir = outputDataDirectory.OutputDataDirectory(local_scenario_output_path)
    print("OutputDataDirectory initialized.")

    # Create the Pilates Output Data object, which initializes various OutputDataFrame subclasses
    pilatesData = outputDataDirectory.BeamOutputData(
        outputDataDirectory=outputDir,
        beamRunInputDirectory=beamInputDirectory,
        collectEvents=True,
    )
    print("PilatesOutputData initialized.")

except Exception as e:
    print(f"\nFATAL ERROR during initialization: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)


print("\n--- Testing Access to Aggregated Output DataFrames ---")

# List of attributes on pilatesData that are OutputDataFrame subclasses
# These represent the aggregated results across years or iterations

"""

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
            self.inputDirectory.linkStatsFile(self.inputDirectory.numberOfIterations),
            self.labeledNetwork,
        )
        self.networkVolumesByLinkByIteration = NetworkVolumesByLinkByIteration(
            self.outputDataDirectory,
            self.inputDirectory,
            self.labeledNetwork,
            list(range(self.inputDirectory.numberOfIterations)),
        )
"""


output_attributes_to_test = [
    "realizedModeCount",
    "modeVMT",
    "modeVHT",
    "passengerMilesByVehicleAndMode",
    "modeEnergy",
    "modePMT",
    "replanningEventReasons",
    "scoreStats",
    # 'linkStatsFromPathTraversals',
    # 'labeledNetwork',
    # 'labeledLinkStatsFile',
    "tazTrafficVolumes",
    "networkVolumesByLink",
    "networkVolumesByLinkByIteration",
]

for attr_name in output_attributes_to_test:
    print(f"\nTesting '{attr_name}'...")
    try:
        # Get the OutputDataFrame object using getattr
        output_obj = getattr(pilatesData, attr_name)

        # Access the dataFrame property to trigger loading/processing
        df = output_obj.dataFrame

        # Check the result
        if df is not None:
            if not df.empty:
                print(f"  SUCCESS: Loaded '{attr_name}'. Shape: {df.shape}")
                # Attempt to save to CSV
                print(df)
                try:
                    output_obj.toCsv()
                    print(f"  Saved '{attr_name}' to CSV.")
                except Exception as save_e:
                    print(f"  ERROR saving '{attr_name}' to CSV: {save_e}")
                    # traceback.print_exc() # Uncomment for detailed save errors

            else:
                print(f"  INFO: Loaded '{attr_name}' is empty.")
        else:
            print(f"  WARNING: Loading '{attr_name}' returned None.")

    except AttributeError:
        print(f"  SKIPPING: Attribute '{attr_name}' not found on PilatesOutputData.")
    except Exception as e:
        print(f"  ERROR processing '{attr_name}': {e}")
        print("  Traceback:")
        traceback.print_exc()

print("\n--- Test Script Finished ---")
