from src import input, outputDataDirectory


if __name__ == "__main__":
    asimFolderName = "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231211/activitysim/year-2010-iteration--1"
    directory = input.ActivitySimRunInputDirectory(asimFolderName)
    asimOutputData = outputDataDirectory.ActivitySimOutputData(outputDataDirectory.OutputDataDirectory("output"), directory, None)
    asimOutputData.mandatoryLocationsByTaz.toCsv()


    pilatesFolderName = "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231107"
    directory = input.PilatesRunInputDirectory(pilatesFolderName, [2010, 2015], 2)
    pilatesData = outputDataDirectory.PilatesOutputData(outputDataDirectory.OutputDataDirectory("output"), directory)




    # baseFolderName = "https://storage.googleapis.com/beam-core-outputs/output/sf-light/sf-light-1k-xml__2022-11-30_15-11-36_zlb"
    baseFolderName = "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231107/beam/year-2015-iteration-2"
    directory = input.BeamRunInputDirectory(
        baseFolderName,
        # "https://storage.googleapis.com/beam-core-outputs/output/sf-light/sf-light-1k-xml__2022-11-30_15-11-36_zlb",
        # "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231107/beam/year-2015-iteration-2",
        0,
    )
    beamOutputData = outputDataDirectory.BeamOutputData(
        outputDataDirectory.OutputDataDirectory("output"), directory
    )
    beamOutputData.modeVMT.toCsv()
    print("stop")
