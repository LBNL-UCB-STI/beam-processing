import os
import logging
from pathlib import Path

from src.input_base import OutputDirectory

# Set up logging
logger = logging.getLogger(__name__)


class OutputDataDirectory:
    """
    Represents an output data directory where results of postprocessing will be saved.

    Attributes:
        path (str): The path to the output data directory.
    """

    def __init__(self, path: str):
        if not path:
            raise ValueError("Output directory path cannot be empty")

        self.path = str(Path(path).resolve())  # Normalize path

        # Create directory if it doesn't exist
        try:
            os.makedirs(self.path, exist_ok=True)
            logger.info(f"Output data directory initialized: {self.path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {self.path}: {e}")
            raise

    def is_writable(self) -> bool:
        """Check if the directory is writable."""
        return os.access(self.path, os.W_OK)

    def get_file_path(self, filename: str) -> str:
        """Get the full path for a file in this directory."""
        return os.path.join(self.path, filename)


class ModelOutputData:
    """
    Base class for model output data containers.

    Attributes:
        outputDataDirectory (OutputDataDirectory): Directory for processed output
        inputDirectory (OutputDirectory): Directory containing raw input data
        remoteResults (bool): Whether results are stored remotely
    """

    def __init__(self, outputDataDirectory: OutputDataDirectory, inputDirectory: OutputDirectory):
        if not isinstance(outputDataDirectory, OutputDataDirectory):
            raise TypeError("outputDataDirectory must be an OutputDataDirectory instance")
        if not isinstance(inputDirectory, OutputDirectory):
             # This check might be too strict if inputDirectory is a subclass,
             # but it catches basic type mismatches.
             logger.warning(f"inputDirectory is not an OutputDirectory instance, but type {type(inputDirectory)}")

        self.outputDataDirectory = outputDataDirectory
        self.inputDirectory = inputDirectory
        # Safely access isLink attribute
        self.remoteResults = getattr(inputDirectory, 'isLink', False)

        logger.info(f"ModelOutputData initialized with input: {getattr(inputDirectory, 'directoryPath', 'Unknown')}, "
                   f"output: {outputDataDirectory.path}, remote: {self.remoteResults}")

    def validate_directories(self) -> bool:
        """Validate that both input and output directories are accessible."""
        try:
            # Check output directory
            if not self.outputDataDirectory.is_writable():
                logger.error(f"Output directory is not writable: {self.outputDataDirectory.path}")
                return False

            # Check input directory if it's local
            # Assume remote directories don't need local path validation
            if not self.remoteResults and hasattr(self.inputDirectory, 'directoryPath'):
                input_path = self.inputDirectory.directoryPath
                if not os.path.exists(input_path):
                    logger.error(f"Input directory does not exist: {input_path}")
                    return False
                # Optional: Check if input directory is readable
                if not os.access(input_path, os.R_OK):
                     logger.error(f"Input directory is not readable: {input_path}")
                     return False


            return True

        except Exception as e:
            logger.error(f"Error validating directories: {e}")
            return False
