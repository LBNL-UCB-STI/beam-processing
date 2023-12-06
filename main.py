from src import input, output

if __name__ == "__main__":
    # baseFolderName = "https://storage.googleapis.com/beam-core-outputs/output/sf-light/sf-light-1k-xml__2022-11-30_15-11-36_zlb"
    baseFolderName = "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231107/beam/year-2015-iteration-2"
    directory = input.BeamRunInputDirectory(
        baseFolderName,
        # "https://storage.googleapis.com/beam-core-outputs/output/sf-light/sf-light-1k-xml__2022-11-30_15-11-36_zlb",
        # "https://storage.googleapis.com/beam-core-outputs/sfbay-demos-base-20231107/beam/year-2015-iteration-2",
        0,
    )
    beamOutputData = output.BeamOutputData(
        output.OutputDataDirectory("output"), directory
    )
    beamOutputData.modeVMT.toCsv()
    print("stop")
