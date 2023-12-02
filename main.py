from src import input, output

if __name__ == "__main__":
    directory = input.BeamRunInputDirectory(
        "https://storage.googleapis.com/beam-core-outputs/output/sf-light/sf-light-1k-xml__2022-11-30_15-11-36_zlb",
        0,
    )
    beamOutputData = output.BeamOutputData(
        output.OutputDataDirectory("output"), directory
    )
    beamOutputData.modeVMT.toCsv()
    print("stop")
