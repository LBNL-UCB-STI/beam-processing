from src import input, outputDataDirectory


if __name__ == "__main__":
    # asimFolderName = "https://storage.googleapis.com/beam-core-outputs/austin-2010-base-20221026/activitysim/year-2010-iteration--1"
    # directory = input.ActivitySimRunInputDirectory(asimFolderName)
    # asimOutputData = outputDataDirectory.ActivitySimOutputData(outputDataDirectory.OutputDataDirectory("output"), directory, None)
    # asimOutputData.mandatoryLocationsByTaz.toCsv()

    pilatesFolderName = (
        "https://storage.googleapis.com/beam-core-outputs/austin-2010-base-20221026"
    )
    directory = input.PilatesRunInputDirectory(pilatesFolderName, [2010, 2015, 2017], 3)
    pilatesData = outputDataDirectory.PilatesOutputData(
        outputDataDirectory.OutputDataDirectory("output"), directory
    )

    processedPersonTrips = pilatesData.runInexus(2017, 2)

