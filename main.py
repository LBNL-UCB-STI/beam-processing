from src import input, outputDataDirectory


if __name__ == "__main__":

    pilatesFolderName = (
        "https://storage.googleapis.com/beam-core-outputs/austin-2010-base-20221026"
    )
    directory = input.PilatesRunInputDirectory(pilatesFolderName, [2010, 2015, 2017], 3)
    pilatesData = outputDataDirectory.PilatesOutputData(
        outputDataDirectory.OutputDataDirectory("output"), directory
    )

    pilatesData.congestionInfoByYear.toCsv()

    pilatesData.beamRuns[(2017, 2)].tazTrafficVolumes.toCsv()
    pilatesData.asimRuns[(2017, 2)].tripPMTByPrimaryPurpose.toCsv()

    processedPersonTrips = pilatesData.runInexus(2017, 2)

