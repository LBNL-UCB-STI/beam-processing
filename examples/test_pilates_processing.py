import os
import sys
import traceback  # Import traceback for detailed error logging

import src.output_container
import src.pilates_output_container
from src.input_directories import PilatesRunOutputDirectory

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
PILATES_OUTPUT_PATH = "output/test-pilates-run"

# Years present in the specified Pilates run
# Based on checking the GCS path, years seem to be [2010, 2012, 2014, 2016, 2018, 2020]
PILATES_YEARS = [2018]

# Number of ActivitySim Light iterations (last iteration used for ASIM outputs)
ASIM_LITE_ITERATIONS = (
    0  # Assuming 2 iterations based on directory structure year-YYYY-iteration-1/2
)

# Number of BEAM iterations (last iteration used for BEAM outputs like linkstats)
# Note: The specified path contains only iteration 0 for BEAM.
BEAM_ITERATIONS = 0  # Assuming only iteration 0 is present

# Region for geometry loading
REGION = "SFBay"

# Local directory to save processed outputs and cache
LOCAL_OUTPUT_BASE_PATH = "./processed_outputs_test"

# Target year and iteration for testing runInexus
# Choose a year and iteration that exist in the data
TARGET_INEXUS_YEAR = 2019  # Must be in PILATES_YEARS
TARGET_INEXUS_ITER = 0  # Must be a valid BEAM iteration for that year in the input path

# --- Script Execution ---

# Set GCLOUD_PROJECT environment variable for GCS access
# Replace with your Google Cloud project ID if necessary
os.environ["GCLOUD_PROJECT"] = "1010663794916"
print(f"Set GCLOUD_PROJECT to {os.environ['GCLOUD_PROJECT']}")
os.chdir("../")

# Ensure the local output directory exists
local_scenario_output_path = os.path.join(LOCAL_OUTPUT_BASE_PATH, "sfbay_test_run")
os.makedirs(local_scenario_output_path, exist_ok=True)
print(f"Local output directory: {local_scenario_output_path}")


print("\n--- Initializing Pilates Input and Output Data ---")
try:
    # Create the Pilates Input Directory object
    pilatesInputDirectory = PilatesRunOutputDirectory(
        PILATES_OUTPUT_PATH,
        years=PILATES_YEARS,
        asimLiteIterations=ASIM_LITE_ITERATIONS,
        beamIterations=BEAM_ITERATIONS,
        region=REGION,
        # Set collectEvents=False to avoid loading ALL events upfront, let OutputDataFrames handle it
        collectEvents=False,
        file_format="parquet",
    )
    print("PilatesRunInputDirectory initialized.")

    # Create the OutputDataDirectory object for saving processed outputs
    outputDir = src.output_container.OutputDataDirectory(local_scenario_output_path)
    print("OutputDataDirectory initialized.")

    # Create the Pilates Output Data object, which initializes various OutputDataFrame subclasses
    pilatesData = src.pilates_output_container.PilatesOutputData(
        outputDataDirectory=outputDir,
        pilatesRunInputDirectory=pilatesInputDirectory,
        region=REGION,  # Pass region again
        # Set collectEvents=False here as well, consistent with InputDirectory
        collectEvents=False,
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
    "tripModeCountPerYear",
    "tourModeCountPerYear",
    "tripModeCountPerIteration",
    "tourModeCountPerIteration",
    "tripModeCountByCountyPerYear",
    "replanningEventReasonPerIteration",
    "scoreStatsByIteration",
    "modeVMTPerYear",
    "modeEnergyPerYear",
    "modePMTPerYear",
    "modePMTPerIteration",
    "congestionInfoByYear",
    "congestionInfoByIteration",
    "passengerMilesByVehicleAndModeByYear",
    "passengerMilesByVehicleAndModeByIteration",
    "realizedModeCountyByIteration",
    "mandatoryLocationsByTazByYear",
    "tripPMTPerYear",
    "tripPMTByPrimaryPurposePerYear",
    "tripPMTByCountyPerYear",
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


print("\n--- Testing runInexus for a Specific Run ---")

# Check if the target year and iteration exist in the loaded data
if (TARGET_INEXUS_YEAR, TARGET_INEXUS_ITER) in pilatesData.beamRuns and (
    TARGET_INEXUS_YEAR,
    TARGET_INEXUS_ITER,
) in pilatesData.asimRuns:

    print(
        f"Attempting runInexus for Year {TARGET_INEXUS_YEAR}, Iteration {TARGET_INEXUS_ITER}..."
    )
    try:
        # Call runInexus on the PilatesOutputData object
        aggregated_trips_df = pilatesData.runInexus(
            TARGET_INEXUS_YEAR, TARGET_INEXUS_ITER
        )

        if aggregated_trips_df is not None:
            if not aggregated_trips_df.empty:
                print(
                    f"  SUCCESS: runInexus completed. Result shape: {aggregated_trips_df.shape}"
                )
                # Save the result to CSV
                inexus_output_path = os.path.join(
                    local_scenario_output_path,
                    f"inexus_year{TARGET_INEXUS_YEAR}_iter{TARGET_INEXUS_ITER}.csv",
                )
                try:
                    aggregated_trips_df.to_csv(inexus_output_path, index=False)
                    print(f"  Saved Inexus output to {inexus_output_path}")
                except Exception as save_e:
                    print(f"  ERROR saving Inexus output to CSV: {save_e}")
                    # traceback.print_exc()

            else:
                print(
                    f"  INFO: runInexus completed but returned an empty DataFrame for Year {TARGET_INEXUS_YEAR}, Iteration {TARGET_INEXUS_ITER}."
                )
        else:
            print(
                f"  WARNING: runInexus returned None for Year {TARGET_INEXUS_YEAR}, Iteration {TARGET_INEXUS_ITER}."
            )

    except Exception as e:
        print(
            f"  ERROR during runInexus for Year {TARGET_INEXUS_YEAR}, Iteration {TARGET_INEXUS_ITER}: {e}"
        )
        print("  Traceback:")
        traceback.print_exc()

else:
    print(
        f"\nSKIPPING runInexus test: Specified year/iteration ({TARGET_INEXUS_YEAR}, {TARGET_INEXUS_ITER}) not found in loaded Beam or ASim runs."
    )


print("\n--- Test Script Finished ---")
