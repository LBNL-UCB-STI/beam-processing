from src import input, outputDataDirectory
import os

os.environ["GCLOUD_PROJECT"] = "1010663794916"

# Ensure the output directory exists
output_base_path = "output/quickstart_run"
os.makedirs(output_base_path, exist_ok=True)

#
# # Example 1: Processing a single BEAM run
# print("--- Processing single BEAM run ---")
# # Replace with an actual BEAM run path (can be local or gs://)
# beamFolderName = "gs://beam-core-outputs/output/sfbay/sfbay-freight-base2018-calib-X__2023-12-14_03-25-14_dlx"
# # Specify number of iterations (usually the last one for linkstats)
# # and region for geometry
# beamInputDirectory = input.BeamRunInputDirectory(beamFolderName, numberOfIterations=15, region="SFBay")
#
# # Create the output data object for this BEAM run
# beamOutputData = outputDataDirectory.BeamOutputData(
#     outputDataDirectory.OutputDataDirectory(os.path.join(output_base_path, "beam_run")),
#     beamInputDirectory
# )
#
# # Access and save some outputs
# print("Saving TAZ Traffic Volumes...")
# # tazTrafficVolumes depends on LabeledLinkStatsFile, which depends on LinkStatsFile and LabeledNetwork
# # Accessing .dataFrame will trigger loading and processing if not cached.
# taz_volumes = beamOutputData.tazTrafficVolumes.dataFrame
# if taz_volumes is not None and not taz_volumes.empty:
#     beamOutputData.tazTrafficVolumes.toCsv()
#     print("TAZ Traffic Volumes saved.")
# else:
#     print("No TAZ Traffic Volumes data found.")
#
# print("\nSaving Mode VMT...")
# mode_vmt = beamOutputData.modeVMT.dataFrame
# if mode_vmt is not None and not mode_vmt.empty:
#     beamOutputData.modeVMT.toCsv()
#     print("Mode VMT saved.")
# else:
#     print("No Mode VMT data found.")


# Example 2: Processing a Pilates run (multiple years/iterations of ASim and BEAM)
print("\n--- Processing Pilates run ---")
# Replace with an actual Pilates run path (can be local or gs://)
# Note: Pilates runs contain ASIM output within subfolders like 'activitysim/output/year-YYYY-iteration-I'
# and BEAM output within 'beam/beam_output/REGION/year-YYYY-iteration-I'
pilatesFolderName = "gs://beam-core-outputs/seattle-util-diff-20240715"  # Example path
# Specify the years, ASIM iterations (usually the last one matters for ASIM outputs),
# and BEAM iterations (last one matters for BEAM outputs).
pilatesInputDirectory = input.PilatesRunInputDirectory(
    pilatesFolderName,
    years=[2010, 2012, 2014, 2016, 2018, 2020],
    asimLiteIterations=2,
    beamIterations=0,
    region="Seattle",
)

# Create the output data object for this Pilates run
pilatesData = outputDataDirectory.PilatesOutputData(
    outputDataDirectory.OutputDataDirectory(
        os.path.join(output_base_path, "pilates_run")
    ),
    pilatesInputDirectory,
    region="Seattle",  # Ensure region is specified consistently
)

# Access and save some outputs from the Pilates run aggregation classes
print("\nSaving Mandatory Locations By TAZ By Year...")
mand_locs = pilatesData.mandatoryLocationsByTazByYear.dataFrame
if mand_locs is not None and not mand_locs.empty:
    pilatesData.mandatoryLocationsByTazByYear.toCsv()
    print("Mandatory Locations By TAZ By Year saved.")
else:
    print("No Mandatory Locations By TAZ By Year data found.")


print("\nSaving Trip PMT By Year...")
trip_pmt_year = pilatesData.tripPMTPerYear.dataFrame
if trip_pmt_year is not None and not trip_pmt_year.empty:
    pilatesData.tripPMTPerYear.toCsv()
    print("Trip PMT By Year saved.")
else:
    print("No Trip PMT By Year data found.")


# Example of running Inexus processing for a specific year/iteration
print("\n--- Running Inexus processing for a specific year/iteration ---")
target_year = 2017
target_iter = 3  # Use the last iteration for ASIM typically

# The runInexus method is now called on the PilatesOutputData object,
# but the actual processing logic is delegated to the BeamOutputData instance for that run.
# It returns the final aggregated trip DataFrame.
print(f"Running Inexus for year {target_year}, iteration {target_iter}")
processedPersonTrips_2017_3 = pilatesData.runInexus(target_year, target_iter)

if processedPersonTrips_2017_3 is not None and not processedPersonTrips_2017_3.empty:
    print(
        f"Inexus processing complete. Result shape: {processedPersonTrips_2017_3.shape}"
    )
    # Save the resulting aggregated trip DataFrame
    output_file_path = os.path.join(
        pilatesData.outputDataDirectory.path,
        f"inexus_year{target_year}_iter{target_iter}.csv",
    )
    print(f"Saving processed person trips to {output_file_path}...")
    processedPersonTrips_2017_3.to_csv(
        output_file_path, index=False
    )  # Save without index for final table
    print("Processed person trips saved.")
else:
    print(
        f"Inexus processing failed or returned no data for year {target_year}, iteration {target_iter}."
    )


# Example of accessing data aggregated across iterations for a year (e.g. 2017, iter 1, 2, 3)
print("\nSaving Trip Mode Count By Iteration...")
trip_mode_count_iter = pilatesData.tripModeCountPerIteration.dataFrame
if trip_mode_count_iter is not None and not trip_mode_count_iter.empty:
    pilatesData.tripModeCountPerIteration.toCsv()
    print("Trip Mode Count By Iteration saved.")
else:
    print("No Trip Mode Count By Iteration data found.")

# Example of accessing score stats by iteration
print("\nSaving Score Stats By Iteration...")
score_stats_iter = pilatesData.scoreStatsByIteration.dataFrame
if score_stats_iter is not None and not score_stats_iter.empty:
    pilatesData.scoreStatsByIteration.toCsv()
    print("Score Stats By Iteration saved.")
else:
    print("No Score Stats By Iteration data found.")

print("\nExample script finished.")

# pilatesData.congestionInfoByYear.toCsv() # Example commented out
# pilatesData.asimRuns[(2017, 3)].tripPMTByPrimaryPurpose.toCsv() # Example commented out

# # Example of accessing data from a specific ASim run directly
# print("\nSaving Trip PMT by Primary Purpose for Year 2017, Iteration 3...")
# try:
#      asim_run_2017_3 = pilatesData.asimRuns[(2017, 3)]
#      trip_pmt_2017_3 = asim_run_2017_3.tripPMTByPrimaryPurpose.dataFrame
#      if trip_pmt_2017_3 is not None and not trip_pmt_2017_3.empty:
#           asim_run_2017_3.tripPMTByPrimaryPurpose.toCsv() # Saves to the asim_run specific output subfolder
#           print("Trip PMT by Primary Purpose for Year 2017, Iteration 3 saved.")
#      else:
#           print("No Trip PMT by Primary Purpose data found for Year 2017, Iteration 3.")
# except KeyError:
#      print("ASIM run for Year 2017, Iteration 3 not found.")
