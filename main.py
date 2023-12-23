from src import input, outputDataDirectory


if __name__ == "__main__":
    beamFolderName = "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-freight-base2018-calib-X__2023-12-14_03-25-14_dlx"
    beamDirectory = input.BeamRunInputDirectory(beamFolderName, numberOfIterations=15, region="SFBay")
    beamData = outputDataDirectory.BeamOutputData(outputDataDirectory.OutputDataDirectory("output/beam-2018-calibration"), beamDirectory)
    beamData.tazTrafficVolumes.toCsv()


    pilatesFolderName = (
        "https://storage.googleapis.com/beam-core-outputs/austin-2010-base-20221026"
    )
    directory = input.PilatesRunInputDirectory(
        pilatesFolderName, [2010, 2015, 2017], 3, region="Austin"
    )
    pilatesData = outputDataDirectory.PilatesOutputData(
        outputDataDirectory.OutputDataDirectory("output/ausin-2010-base"), directory, region="Austin"
    )

    pilatesData.congestionInfoByYear.toCsv()
    pilatesData.asimRuns[(2017, 3)].tripPMTByPrimaryPurpose.toCsv()

    processedPersonTrips = pilatesData.runInexus(2017, 2)
