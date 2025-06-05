from src.input_base import InputDirectory


class OutputDataDirectory:
    """
    Represents an output data directory where results of postprocessing will be saved.

    Attributes:
        path (str): The path to the output data directory.
    """

    def __init__(self, path):
        self.path = path


class ModelOutputData:
    def __init__(
        self, outputDataDirectory: OutputDataDirectory, inputDirectory: InputDirectory
    ):
        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        self.remoteResults = inputDirectory.isLink
