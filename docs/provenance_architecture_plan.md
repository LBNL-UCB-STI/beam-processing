# Model Run Provenance Tracking Architecture Plan

## Goal

The primary goal is to implement a system that tracks the lineage of model runs, specifically:
1.  For any given model run, identify all the specific model outputs that served as its inputs.
2.  For any given model output, identify the specific model run that produced it.

This system will eventually integrate with a central database to store this provenance information, allowing for comprehensive tracing of data flow through the simulation pipeline.

## Core Concepts

*   **Run ID:** A unique identifier assigned to each execution of the model pipeline. This ID will link all inputs consumed and outputs produced by that specific run.
*   **Input Provenance:** The ability to determine which previous run produced a specific input file used by the current run.
*   **Output Tagging:** Associating each output file produced by the current run with the current run's unique ID.
*   **Central Registry (Database):** A persistent storage mechanism (like a relational database) to record the relationships between runs, inputs, and outputs.

## Proposed Architecture Components

1.  **`RunContext` Object:**
    *   A new object instantiated at the very beginning of a model run.
    *   Holds the unique `run_id` for the current execution.
    *   Acts as a central point for interacting with the provenance tracking system (initially, this might be simple print statements or logging, later database calls).
    *   Stores metadata about the current run (e.g., start time, parameters, code version).

2.  **Input Identification Mechanism:**
    *   Input data is currently represented by `OutputDirectory` and `RawOutputFile` (and indirectly by `ProcessedDataFrame` which loads `RawOutputFile`s).
    *   These input objects need a way to determine the `run_id` of the run that *produced* the data they are loading.
    *   **Proposed Mechanism:** Adopt a convention where output data directories include the `run_id` of the producing run in their path (e.g., `/path/to/outputs/<source_run_id>/<filename>`). Input objects can parse this `source_run_id` from their `directoryPath` or `filePath`.

3.  **Output Tagging Mechanism:**
    *   Output data is currently managed by `OutputDataDirectory` and produced by `ProcessedDataFrame`.
    *   The `OutputDataDirectory` instance used for saving outputs must be aware of the current run's `run_id`.
    *   **Proposed Mechanism:** Pass the current `run_id` (from the `RunContext`) to the `OutputDataDirectory` constructor. The `OutputDataDirectory` will then incorporate this `run_id` into the paths it generates for saving files (e.g., `/path/to/processed_data/<current_run_id>/<output_type>/<filename>`).

4.  **Database Interaction Layer:**
    *   The `RunContext` object will contain methods to interact with the database (or a database abstraction layer).
    *   **Run Start:** Record the new `run_id` and initial metadata in a `ModelRuns` table.
    *   **Input Recording:** When an input file is successfully loaded by a `RawOutputFile` or `ProcessedDataFrame`, notify the `RunContext`. The `RunContext` will record a link in a `ModelInputs` table between the current `run_id` and the `source_run_id` and `file_path` of the input.
    *   **Output Recording:** When a `ProcessedDataFrame` saves its data, notify the `RunContext`. The `RunContext` will record a link in a `ModelOutputs` table with the current `run_id`, the type of output (e.g., 'TripPMT'), and the saved `file_path`.
    *   **Run End:** Update the `ModelRuns` record with the end time and final status.

## Database Schema (Conceptual)

*   **`ModelRuns` Table:**
    *   `run_id` (Primary Key, e.g., UUID)
    *   `start_time` (Timestamp)
    *   `end_time` (Timestamp, nullable)
    *   `status` (e.g., 'running', 'completed', 'failed')
    *   `parameters` (JSON or text, storing run configuration)
    *   `code_version` (e.g., Git hash)
    *   `hostname` (Machine where run occurred)
    *   ... other relevant metadata

*   **`ModelOutputs` Table:**
    *   `output_id` (Primary Key, e.g., UUID)
    *   `run_id` (Foreign Key to `ModelRuns.run_id`)
    *   `output_type` (e.g., 'ProcessedPersonsFile', 'TripPMTByYear')
    *   `file_path` (Path to the output file)
    *   `created_time` (Timestamp)
    *   ... other relevant metadata

*   **`ModelInputs` Table:**
    *   `input_id` (Primary Key, e.g., UUID)
    *   `run_id` (Foreign Key to `ModelRuns.run_id`, the run *consuming* the input)
    *   `source_output_id` (Foreign Key to `ModelOutputs.output_id`, the specific output being consumed)
    *   `consumed_time` (Timestamp)
    *   ... other relevant metadata

*(Note: The `ModelInputs` table linking directly to `ModelOutputs` requires looking up the `output_id` based on the `source_run_id` and `file_path` when an input is loaded. An alternative is to store `source_run_id` and `source_file_path` directly in `ModelInputs` and link later, or query `ModelOutputs` to find the `source_output_id`.)*

## Step-by-Step Implementation Plan (Code Changes)

*(Note: These steps outline the necessary code modifications. We will address them one by one when requested.)*

1.  **Define `RunContext` Class:**
    *   Create a new class, perhaps in a new file like `src/provenance.py`.
    *   It will have an `__init__` method that generates or accepts a `run_id` (e.g., using `uuid.uuid4()`).
    *   Add placeholder methods like `record_input(source_run_id: str, file_path: str)`, `record_output(output_type: str, file_path: str)`, `record_run_start(...)`, `record_run_end(...)`. Initially, these can just print or log the information.
    *   Consider making it a singleton or passing the instance explicitly. Explicit passing is generally clearer for dependency injection.

2.  **Modify `OutputDataDirectory` (`src/output_container.py`):**
    *   Add a `run_id` parameter to the `__init__` method. Store this `run_id` as an attribute.
    *   Modify the `get_file_path(filename: str)` method to include `self.run_id` in the generated path structure (e.g., `os.path.join(self.path, self.run_id, filename)` or `os.path.join(self.path, self.run_id, self.__class__.__name__, filename)` for more structure).

3.  **Modify Input Classes (`src/input_base.py`, `src/activitysim/activitysim_output_files.py`):**
    *   **`OutputDirectory` (Base Class):** Add an optional `source_run_id` attribute to its `__init__`. Add a property or method to expose this `source_run_id`. Implement logic to try and parse the `source_run_id` from `directoryPath` if not provided explicitly (based on the new output path convention).
    *   **`RawOutputFile` (Base Class):** Inherits from `OutputDirectory` or is associated with one. Ensure it can access the `source_run_id` of its parent directory. Add a property to expose its own `source_run_id` and `filePath`.
    *   **Specific `RawOutputFile` Subclasses (e.g., in `activitysim_output_files.py`):** Update their `__init__` methods if necessary to handle the `source_run_id` parameter or ensure they correctly inherit/access it from the `inputDirectory`.

4.  **Modify `ProcessedDataFrame` (`src/processed_data_frame.py`):**
    *   Add a `run_context: RunContext` parameter to its `__init__` method. Store this instance.
    *   Modify the `load()` method: After successfully loading data from its source (`self.inputDirectory` which is a `RawOutputFile` or similar), call `self.run_context.record_input(source_run_id, file_path)`. This requires the source object to expose these details.
    *   Modify the `save()` method (or the internal caching logic that handles saving): Before saving the DataFrame to `self._diskLocation`, call `self.run_context.record_output(self.__class__.__name__, self._diskLocation)`.

5.  **Modify `ModelOutputData` (`src/output_container.py`) and other top-level containers:**
    *   These classes are responsible for orchestrating the creation of `ProcessedDataFrame` instances.
    *   Add a `run_context: RunContext` parameter to their `__init__`.
    *   When creating `OutputDataDirectory` instances, pass the `run_context.run_id`.
    *   When creating `ProcessedDataFrame` instances, pass the `run_context` instance.
    *   Ensure that input directories (`ActivitySimRunOutputDirectory`, `PilatesRunOutputDirectory`) are initialized such that their `source_run_id` can be determined (either by parsing the path or by passing it explicitly if known).

6.  **Integrate `RunContext` in Main Execution Script:**
    *   At the very entry point of your model execution script, create the `RunContext` instance.
    *   Pass this `RunContext` instance down through the initialization of the main data processing objects (`ModelOutputData`, etc.).
    *   Call `run_context.record_run_start(...)` at the beginning and `run_context.record_run_end(...)` at the end (handling success/failure).

## Considerations

*   **Database Implementation:** The plan assumes a database backend. The initial implementation can use simple logging or a JSON file to record provenance before connecting to a real database.
*   **Input Identification Complexity:** Parsing `source_run_id` from paths requires a strict and consistent directory naming convention for outputs. If inputs can come from arbitrary locations, explicit passing of `source_run_id` might be necessary.
*   **Granularity:** Decide the level of granularity for inputs/outputs (e.g., the raw file, the processed DataFrame, or specific aggregations). The current plan focuses on `ProcessedDataFrame` instances as the trackable units.
*   **Performance:** Database interactions should be efficient, perhaps using batch inserts or asynchronous operations for high-volume runs.
*   **Error Handling:** Ensure provenance recording doesn't cause the main model run to fail. Database errors should be logged but not necessarily stop the simulation.

This plan provides a roadmap for adding provenance tracking. We can now proceed with implementing these steps in the code when you are ready.
