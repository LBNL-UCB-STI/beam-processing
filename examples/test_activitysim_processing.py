import os
import sys
import traceback  # Import traceback for detailed error logging

import src.activitysim.activitysim_output_container
import src.output_container
from src.geometry import SfBayGeometry, SeattleGeometry
from src.activitysim.activitysim_processed_data_frame import ProcessedSkimsFile

# Add the parent directory of src to the Python path
# This assumes the script is run from the project root directory (beam-processing/)
# If running from a different directory, adjust the path accordingly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import modules from src
try:
    from src import input_directories, analysis
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Please ensure the script is run from the beam-processing directory")
    print(f"or that the parent directory '{parent_dir}' is in your Python path.")
    sys.exit(1)


# --- Configuration ---
# Google Cloud Storage path for the Pilates run
# !! NOTE: Ensure you have authenticated to Google Cloud Storage (e.g., `gcloud auth application-default login`)
#    or have the GOOGLE_APPLICATION_CREDENTIALS environment variable set.
ASIM_OUTPUT_PATH = "output/activitysim/sfbay/year-2018-iteration--1"


# Region for geometry loading
REGION = "SfBay"

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
    asimInputDirectory = input.ActivitySimRunInputDirectory(
        ASIM_OUTPUT_PATH, geometry=SfBayGeometry(), file_format="parquet"
    )
    print("AsimRunInputDirectory initialized.")

    # Create the OutputDataDirectory object for saving processed outputs
    outputDir = src.output_container.OutputDataDirectory(local_scenario_output_path)
    print("OutputDataDirectory initialized.")

    # Create the Pilates Output Data object, which initializes various OutputDataFrame subclasses
    pilatesData = src.activitysim.activitysim_output_container.ActivitySimOutputData(
        outputDataDirectory=outputDir,
        activitySimRunInputDirectory=asimInputDirectory,
        geometry=SfBayGeometry(),
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

output_attributes_to_test = [
    "mandatoryLocationsByTaz",
    "tripModeCount",
    "tourModeCount",
    "tripModeCountByOrigin",
    "tripModeCountByPrimaryPurpose",
    # 'tripsByYear', # tripsByYear's accessor calls getAggregatedTrips which is tested below
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
