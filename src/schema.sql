-- Table for model runs
CREATE TABLE model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID
    model_name TEXT NOT NULL,                -- Name of the model
    config JSON NOT NULL,                    -- JSON of configuration parameters
    inputs JSON NOT NULL,                    -- JSON of input data identifiers
    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP -- When the run was performed
);

-- Table for data outputs
CREATE TABLE data_outputs (
    data_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Unique ID
    data_name TEXT NOT NULL,                   -- Unique name for the data
    run_id INTEGER NOT NULL,                   -- Foreign key to the model run
    metadata JSON,                             -- JSON metadata (e.g., year, iteration)
    UNIQUE(data_name),                         -- Enforce uniqueness
    FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
);
