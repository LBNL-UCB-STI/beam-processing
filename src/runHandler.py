from typing import Dict, Optional, List, Callable, Dict, Any, Union
import matplotlib.pyplot as plt
import networkx as nx
import sqlite3
import json


class GlobalRunState:
    def __init__(self, model_name: Optional[str] = None, year: int = 0, iteration: Optional[int] = None):
        self.model_name = model_name
        self.year = year
        self.iteration = iteration

    def __repr__(self):
        return f"GlobalRunState(model_name={self.model_name}, year={self.year}, iteration={self.iteration})"

    @staticmethod
    def fromRunName(runName) -> 'GlobalRunState':
        parts = runName.split('_')
        name, year, iteration = parts + [None] * (3 - len(parts))
        return GlobalRunState(name, year, iteration)

    def toRunName(self, data_type: Optional[str] = None) -> str:
        name = data_type or self.model_name or 'EMPTY'
        out = [name, str(self.year)]
        if self.iteration is not None:
            out.append(str(self.iteration))
        return "_".join(out)


class Data:
    """Base class for data objects."""

    def __init__(self, data_type: Optional[str], config_or_state: Union['Configuration', GlobalRunState], producer: Optional['ModelRun'] = None):
        if isinstance(config_or_state, GlobalRunState):
            self.config = None
            self.name = config_or_state.toRunName(data_type)
        else:
            self.config = config_or_state
            self.name = config_or_state.state.toRunName(data_type)
        self.producer = producer

        # Register this Data in the global configuration (if available)
        if self.config and producer:
            producer.global_config.register_data(self)

    def __repr__(self):
        return f"Data(name={self.name}, producer={self.producer.name if self.producer else None})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Data):
            return False
        return self.name == other.name

    def update_database(self, db_path):
        """Marks this Data object as existing in the database."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if the data already exists
            query = "SELECT data_name FROM data_outputs WHERE data_name = ?"
            cursor.execute(query, (self.name,))
            if cursor.fetchone():
                print(f"Data {self.name} already exists in the database.")
                return

            # Insert the data into the database
            cursor.execute("""
            INSERT INTO data_outputs (data_name, run_id, metadata)
            VALUES (?, ?, ?);
            """, (self.name, self.producer.run_id if self.producer else None, json.dumps(self.config.parameters)))
            conn.commit()
            print(f"Data {self.name} added to the database.")


class Population(Data):
    """Represents a synthetic population for a specific year."""

    def __init__(self, producer: Optional['ModelRun'], config: 'Configuration'):
        if config is None:
            raise ValueError("Configuration is required to initialize a Population.")
        self.year = config.parameters["year"]  # Extract the year from the Configuration
        super().__init__(data_type="Population", producer=producer, config_or_state=config)


class ActivityPlans(Data):
    """Represents a set of activity plans for a population."""

    def __init__(self, producer: Optional['ModelRun'], config: 'Configuration'):
        super().__init__(data_type="ActivityPlans", producer=producer, config_or_state=config)


class Skims(Data):
    """Represents travel times and costs between origin/destination pairs."""

    def __init__(self, producer: Optional['ModelRun'], config:'Configuration'):
        super().__init__(data_type="Skims", producer=producer, config_or_state=config)


class ModelRun:
    """Base class for all model runs."""
    required_inputs: Dict[str, type] = {}  # Dictionary of input_name -> input_type

    def __init__(self, name: str, inputs: Dict[str, Optional[Data]], config: 'Configuration',
                 global_config: 'GlobalConfiguration'):
        self.name = name  # Simplified name without year/iteration redundancy
        self.inputs = inputs
        self.config = config
        self.outputs: List[Data] = []
        self.global_config = global_config
        self.run_id = None

        # Register this ModelRun in the global configuration
        self.global_config.register_model_run(self)

    def resolve_dependencies(self):
        """Resolve dependencies for all required inputs."""
        for input_name, input_type in self.required_inputs.items():
            if input_name not in self.inputs or self.inputs[input_name] is None:
                # Delegate dependency creation to the subclass
                self.inputs[input_name] = self.build_dependency(input_name, input_type)
                if self.inputs[input_name] is not None and isinstance(self.inputs[input_name].producer, ModelRun):
                    self.inputs[input_name].producer.resolve_dependencies()  # Resolve its dependencies

    def update_database(self):
        db_path = self.global_config.database_path
        """Marks this ModelRun as run and stores its information in the database."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if the model run already exists
            config_json = json.dumps(self.config.parameters)
            inputs_json = json.dumps({k: v.name for k, v in self.inputs.items()})
            query = """
            SELECT run_id 
            FROM model_runs 
            WHERE model_name = ? AND config = ? AND inputs = ?;
            """
            cursor.execute(query, (self.name, config_json, inputs_json))
            result = cursor.fetchone()
            if result:
                print(f"ModelRun {self.name} already exists in the database with run_id {result[0]}.")
                self.run_id = result[0]  # Set the run_id if already exists
                return

            # Insert the model run into the database
            cursor.execute("""
            INSERT INTO model_runs (model_name, config, inputs)
            VALUES (?, ?, ?);
            """, (self.name, config_json, inputs_json))
            self.run_id = cursor.lastrowid  # Store the generated run_id
            conn.commit()
            print(f"ModelRun {self.name} added to the database with run_id {self.run_id}.")

            # Add the outputs of this run to the data_outputs table
            for output in self.outputs:
                output.update_database(db_path)

    def build_dependency(self, input_name: str, input_type: type) -> Optional[Data]:
        """To be implemented by subclasses: builds missing dependencies for a specific input."""
        raise NotImplementedError("Subclasses must implement the build_dependency method.")

    def run(self):
        """Placeholder for subclasses to implement."""
        raise NotImplementedError("Subclasses must implement the run method.")

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, ModelRun):
            return False
        return self.name == other.name


class DemographicEvolution(ModelRun):
    """Advances a Population forward by a defined year increment."""
    required_inputs = {"population": Population, "skims": Skims}

    def build_dependency(self, input_name: str, input_type: type) -> Optional[Data]:
        if input_name == "population" and input_type == Population:
            # Generate the previous year's population
            current_year = self.config.parameters["year"]
            previous_year = current_year - 1
            if previous_year < self.global_config.start_year:
                raise ValueError(f"No Population data available before {self.global_config.start_year}")

            previous_config = self.global_config.generate_config("DemographicEvolution", previous_year)
            previous_run = DemographicEvolution(
                name=f"DemoEvo_{previous_year}",
                inputs={"population": Population(year=previous_year), "skims": Skims(year=previous_year)},
                config=previous_config,
                global_config=self.global_config
            )
            previous_run.run()
            return previous_run.outputs[0]
        elif input_name == "previous_skims" and input_type == Skims:
            current_year = self.config.parameters["year"]
            previous_year = current_year - 1
            n_iterations = self.global_config.iteration_settings["iterations_per_year"]
            skims_config = self.global_config.generate_config("NetworkSimulation", previous_year, n_iterations)
            skims_run = NetworkSimulation(
                name=f"TransportSim_{n_iterations}",
                inputs={"population": None, "activity_plans": None, "previous_skims": None},  # Dependencies resolved
                config=skims_config,
                global_config=self.global_config
            )
            skims_run.resolve_dependencies()
            skims_run.run()
            return skims_run.outputs[0]

    def run(self):
        # Extract the current year from the config
        current_year = self.config.parameters["year"]

        # Use the Skims data as needed (logic can be customized)
        skims_data = self.inputs.get("skims")
        print(f"DemographicEvolution: Using skims data: {skims_data.name if skims_data else 'None'}")

        # Generate configuration for the target year
        next_config = self.global_config.generate_config(self.config.state)

        # Create the output Population for this year
        output_population = Population(producer=self, config=next_config)
        self.outputs.append(output_population)
        print(f"DemographicEvolution: Generated population for {current_year}")
        return output_population


class NetworkSimulation(ModelRun):
    """Simulates the transportation network and produces Skims."""
    required_inputs = {
        "population": Population,
        "activity_plans": ActivityPlans,
        "previous_skims": Skims,
    }

    def build_dependency(self, input_name: str, input_type: type) -> Optional[Data]:
        if input_name == "population" and input_type == Population:
            year = self.config.parameters["year"]
            population_config = self.global_config.generate_config(self.config.state)
            population_run = DemographicEvolution(
                name=f"DemoEvo_{year}",
                inputs={"population": None},  # Dependency will be resolved
                config=population_config,
                global_config=self.global_config
            )
            population_run.resolve_dependencies()
            population_run.run()
            return population_run.outputs[0]
        elif input_name == "previous_skims" and input_type == Skims:
            iteration = self.config.parameters["iteration"] - 1
            if iteration < 0:
                return None  # No previous Skims for the first iteration
            skims_config = self.global_config.generate_config(self.config.state)
            skims_run = NetworkSimulation(
                name=f"TransportSim_{iteration}",
                inputs={"population": None, "activity_plans": None, "previous_skims": None},  # Dependencies resolved
                config=skims_config,
                global_config=self.global_config
            )
            skims_run.resolve_dependencies()
            skims_run.run()
            return skims_run.outputs[0]
        elif input_name == "activity_plans" and input_type == ActivityPlans:
            iteration = self.config.parameters["iteration"]
            year = self.config.parameters["year"]
            if iteration < 0:
                return None  # No previous Skims for the first iteration
            plans_config = self.global_config.generate_config(self.config.state)
            plans_run = ActivityDemand(
                name=f"ActivityDemand_{year}_{iteration}",
                inputs={"population": None, "activity_plans": None, "previous_skims": None},  # Dependencies resolved
                config=plans_config,
                global_config=self.global_config
            )
            plans_run.resolve_dependencies()
            plans_run.run()
            return plans_run.outputs[0]

    def run(self):
        iteration = self.config.parameters.get("iteration", 0)
        output_skims = Skims(producer=self, config=self.config)
        self.outputs.append(output_skims)
        print(f"NetworkSimulation: Produced skims for iteration {iteration}")
        return output_skims


class ActivityDemand(ModelRun):
    """Generates ActivityPlans based on Skims and Population."""
    required_inputs = {
        "skims": Skims,
        "population": Population,
    }

    def build_dependency(self, input_name: str, input_type: type) -> Optional[Data]:
        if input_name == "skims" and input_type == Skims:
            current_iter = self.config.parameters["iteration"]
            if current_iter > 0:
                prev_iter = current_iter - 1
                year = self.config.parameters["year"]
            else:
                prev_iter = self.global_config.iteration_settings["iterations_per_year"]
                year = self.config.parameters["year"] - self.global_config.iteration_settings[
                    "year_increment_after_loops"]
            skims_config = self.global_config.generate_config("NetworkSimulation", year, iteration=prev_iter)
            skims_run = NetworkSimulation(
                name=f"NetworkSimulation_{year}_{iteration}",
                inputs={"population": None, "activity_plans": None, "previous_skims": None},
                config=skims_config,
                global_config=self.global_config
            )
            skims_run.resolve_dependencies()
            skims_run.run()
            return skims_run.outputs[0]
        elif input_name == "population" and input_type == Population:
            year = self.config.parameters["year"]
            population_config = self.global_config.generate_config("DemographicEvolution", year)
            population_run = DemographicEvolution(
                name=f"DemographicEvolution_{year}",
                inputs={"population": None},
                config=population_config,
                global_config=self.global_config
            )
            population_run.resolve_dependencies()
            population_run.run()
            return population_run.outputs[0]
        else:
            print("SDFSDF")

    def run(self):
        iteration = self.config.parameters.get("iteration", 0)
        output_activity_plans = ActivityPlans(producer=self, config=self.config)
        self.outputs.append(output_activity_plans)
        print(f"ActivityDemand: Generated activity plans for iteration {iteration}")
        return output_activity_plans


class GlobalConfiguration:
    def __init__(
            self,
            database_path: str,
            start_year: int,
            end_year: int,
            iteration_settings: Dict[str, Any],
            custom_config_generator: Callable[[GlobalRunState, Dict[str, Any]], Dict[str, Any]],
    ):
        self.database_path: str = database_path
        self.start_year: int = start_year
        self.end_year: int = end_year
        self.iteration_settings: Dict[str, Any] = iteration_settings
        self.custom_config_generator: Callable[
            [GlobalRunState, Dict[str, Any]], Dict[str, Any]] = custom_config_generator

        # Registry to store ModelRun and Data objects
        self.model_registry: Dict[str, ModelRun] = {}
        self.data_registry: Dict[str, Data] = {}

    def generate_config(self, run_state: GlobalRunState) -> 'Configuration':
        """Generates a model configuration dynamically using the custom configuration function."""
        if not self.custom_config_generator:
            raise ValueError("No custom configuration generator function provided.")

        # Generate configuration
        config_parameters = self.custom_config_generator(run_state, self.iteration_settings)
        return Configuration(config_parameters, run_state)

    def check_database(self, data_or_model: Union['Data', 'ModelRun']) -> bool:
        """Checks the database to see if a Data or ModelRun already exists."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()

            if isinstance(data_or_model, Data):
                query: str = "SELECT data_name FROM data_outputs WHERE data_name = ?"
                cursor.execute(query, (data_or_model.name,))
                result = cursor.fetchone()
                return result is not None  # Return True if the data exists

            elif isinstance(data_or_model, ModelRun):
                expected_config: Dict[str, Any] = self.custom_config_generator(
                    state=data_or_model.config.state,
                    iteration_settings=self.iteration_settings,
                )
                expected_config_json: str = json.dumps(expected_config)

                query = """
                    SELECT run_id 
                    FROM model_runs 
                    WHERE model_name = ? AND config = ? AND inputs = ?;
                """
                inputs_json: str = json.dumps({k: v.name for k, v in data_or_model.inputs.items()})
                cursor.execute(query, (data_or_model.name, expected_config_json, inputs_json))
                result = cursor.fetchone()
                return result is not None  # Return True if the model run exists

            else:
                raise ValueError("Unsupported object type for database check.")

    def lookup_model_run(self, name: str, year: int, iteration: Optional[int] = None) -> Optional[ModelRun]:
        """Looks up a ModelRun by name, year, and iteration."""
        key = self._generate_key(name, year, iteration)
        return self.model_registry.get(key)

    def lookup_data(self, name: str, year: int, iteration: Optional[int] = None) -> Optional[Data]:
        """Looks up a Data object by name, year, and iteration."""
        key = self._generate_key(name, year, iteration)
        return self.data_registry.get(key)

    def _generate_key(self, name: str, year: int, iteration: Optional[int]) -> str:
        """Generates a unique key for registry lookup."""
        return f"{name}_{year}_{iteration}" if iteration is not None else f"{name}_{year}"

    def register_model_run(self, model: ModelRun):
        """Registers a ModelRun in the registry."""
        # key = self._generate_key(model.name, model.config.parameters["year"], model.config.parameters.get("iteration"))
        self.model_registry[model.name] = model

    def register_data(self, data: Data):
        """Registers a Data object in the registry."""
        key = data.name  # Use the name already assigned by Data.__init__
        if key not in self.data_registry:
            self.data_registry[key] = data
        else:
            print(f"Skipping {key} because it's already in the registry")

    def get_model_run_from_database(self, model_name: str) -> ModelRun:
        """Retrieves a ModelRun from the database."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT run_id, model_name, config, inputs 
                FROM model_runs 
                WHERE model_name = ?;
            """
            cursor.execute(query, (model_name,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"ModelRun {model_name} not found in the database.")

            run_id, model_name, config_json, inputs_json = row
            config = Configuration(json.loads(config_json), GlobalRunState.fromRunName(model_name))
            inputs = {k: Data(data_type=None, producer=None, config_or_state=GlobalRunState.fromRunName(v)) for k, v in json.loads(inputs_json).items()}

            return ModelRun(name=model_name, inputs=inputs, config=config, global_config=self)

    def run_workflow(self) -> Dict[str, Optional[Data]]:
        """Runs the workflow dynamically, querying the database to avoid redundant runs."""
        workflow_modules: list[str] = self.iteration_settings.get("workflow", [])
        iterations_per_year: int = self.iteration_settings.get("iterations_per_year", 5)
        fixed_inputs: Dict[str, 'Data'] = self.iteration_settings.get("fixed_inputs", {})

        current_state = GlobalRunState(year=self.start_year)
        final_year: int = self.end_year
        current_population: Optional['Data'] = fixed_inputs.get("population")
        current_skims: Optional['Data'] = fixed_inputs.get("skims")
        current_activity_plans: Optional['Data'] = fixed_inputs.get("activity_plans")

        while current_state.year <= final_year:
            print(f"--- Year {current_state.year} ---")

            if "DemographicEvolution" in workflow_modules and current_state.year > self.start_year:
                current_state.model_name = "DemographicEvolution"
                current_state.iteration = 0
                demo_config = self.generate_config(current_state)
                demographic_model = DemographicEvolution(
                    name=f"DemographicEvolution_{current_state.year}",
                    inputs={"population": current_population, "skims": current_skims},
                    config=demo_config,
                    global_config=self,
                )
                if not self.check_database(demographic_model):
                    print(f"DemographicEvolution_{current_state.year} needs to run.")
                    demographic_model.run()
                    current_population = demographic_model.outputs[0]
                else:
                    print(f"DemographicEvolution_{current_state.year} already completed.")
                    # Retrieve the model run from the database
                    producer = self.get_model_run_from_database(demographic_model.name)
                    current_population = Data(
                        data_type="Population",
                        producer=producer,  # Link the producer
                        config_or_state=demo_config,
                    )
                    producer.outputs.append(current_population)

            if "NetworkSimulation" in workflow_modules:
                for iteration in range(iterations_per_year):
                    current_state.model_name = "NetworkSimulation"
                    current_state.iteration = iteration

                    print(f"--- Inner Iteration {iteration + 1} ---")

                    activity_state = GlobalRunState(model_name="ActivityDemand", year=current_state.year,
                                                    iteration=iteration)
                    activity_config = self.generate_config(activity_state)
                    activity_model = ActivityDemand(
                        name=f"ActivityDemand_{current_state.year}_{iteration}",
                        inputs={"skims": current_skims, "population": current_population},
                        config=activity_config,
                        global_config=self,
                    )
                    if not self.check_database(activity_model):
                        print(f"ActivityDemand_{current_state.year}_{iteration} needs to run.")
                        activity_model.run()
                        current_activity_plans = activity_model.outputs[0]
                    else:
                        print(f"ActivityDemand_{current_state.year}_{iteration} already completed.")
                        producer = self.get_model_run_from_database(activity_model.name)
                        current_activity_plans = Data(
                            data_type="ActivityPlans",
                            producer=producer,  # Link the producer
                            config_or_state=activity_config,
                        )
                        producer.outputs.append(current_activity_plans)

                    transport_config = self.generate_config(current_state)
                    transport_model = NetworkSimulation(
                        name=f"NetworkSimulation_{current_state.year}_{iteration}",
                        inputs={
                            "population": current_population,
                            "activity_plans": current_activity_plans,
                            "previous_skims": current_skims,
                        },
                        config=transport_config,
                        global_config=self,
                    )
                    if not self.check_database(transport_model):
                        print(f"NetworkSimulation_{current_state.year}_{iteration} needs to run.")
                        transport_model.run()
                        current_skims = transport_model.outputs[0]
                    else:
                        print(f"NetworkSimulation_{current_state.year}_{iteration} already completed.")
                        producer = self.get_model_run_from_database(transport_model.name)
                        current_skims = Data(
                            data_type="Skims",
                            producer=producer,  # Link the producer
                            config_or_state=transport_config,
                        )
                        producer.outputs.append(current_skims)

            if "DemographicEvolution" in workflow_modules:
                current_state.year += 1
            else:
                break

        print("--- Workflow Completed ---")
        return {
            "final_population": current_population,
            "final_activity_plans": current_activity_plans,
            "final_skims": current_skims,
        }

    def infer_producer(self, data: Data):
        """Infers and creates the producer for a Data object if it lacks one."""
        if "ActivityPlans" in data.name:
            # ActivityPlans is generated by ActivityDemand
            year = data.config.parameters["year"]
            iteration = data.config.parameters.get("iteration", 0)

            # Handle Skims input for iteration 0
            if iteration == 0:
                skims_input = Skims(
                    config=Configuration(
                        {"year": year - 1, "iteration": self.iteration_settings["iterations_per_year"] - 1}),
                    producer=None
                )
            else:
                skims_input = Skims(
                    config=Configuration({"year": year, "iteration": iteration - 1}),
                    producer=None
                )

            activity_config = self.generate_config("ActivityDemand", year=year, iteration=iteration)
            producer = ActivityDemand(
                name=f"ActivityDemand_{year}_{iteration}",
                inputs={
                    "skims": skims_input,
                    "population": Population(config=Configuration({"year": year}), producer=None),
                },
                config=activity_config,
                global_config=self,
            )

            # Link the Data to its producer
            data.producer = producer
            producer.outputs.append(data)
            return producer

        elif "Skims" in data.name:
            # Skims is generated by NetworkSimulation
            year = data.config.parameters["year"]
            iteration = data.config.parameters.get("iteration", 0)

            # Handle previous Skims input for iteration 0
            if iteration == 0:
                previous_skims = Skims(
                    config=Configuration(
                        {"year": year - 1, "iteration": self.iteration_settings["iterations_per_year"] - 1}),
                    producer=None
                )
            else:
                previous_skims = Skims(
                    config=Configuration({"year": year, "iteration": iteration - 1}),
                    producer=None
                )

            transport_config = self.generate_config("NetworkSimulation", year=year, iteration=iteration)
            producer = NetworkSimulation(
                name=f"NetworkSimulation_{year}_{iteration}",
                inputs={
                    "population": Population(config=Configuration({"year": year}), producer=None),
                    "activity_plans": ActivityPlans(config=Configuration({"year": year, "iteration": iteration}),
                                                    producer=None),
                    "previous_skims": previous_skims,
                },
                config=transport_config,
                global_config=self,
            )

            # Link the Data to its producer
            data.producer = producer
            producer.outputs.append(data)
            return producer

        elif "Population" in data.name:
            # Population is generated by DemographicEvolution
            year = data.config.parameters["year"]
            demo_config = self.generate_config("DemographicEvolution", year=year)
            producer = DemographicEvolution(
                name=f"DemographicEvolution_{year}",
                inputs={
                    "population": Population(config=Configuration({"year": year - 1}), producer=None),
                    "skims": Skims(config=Configuration(
                        {"year": year - 1, "iteration": self.iteration_settings["iterations_per_year"] - 1}),
                        producer=None),
                },
                config=demo_config,
                global_config=self,
            )

            # Link the Data to its producer
            data.producer = producer
            producer.outputs.append(data)
            return producer

        raise ValueError(f"Cannot infer producer for Data: {data.name}")

    def build_dependencies(self, obj):
        """Recursively resolves dependencies for a given Data or ModelRun object."""
        if obj in self.processed:
            return  # Skip already-processed objects

        # Mark this object as processed
        self.processed.add(obj)

        if isinstance(obj, Data):
            # If the Data has a producer and the producer is already registered, stop recursion
            if obj.producer:
                if obj.producer.name in self.model_registry:
                    print(f"Data {obj.name} already resolved with producer {obj.producer.name}.")
                    return
                else:
                    # Resolve the producer's dependencies
                    self.build_dependencies(obj.producer)
            else:
                # If Data has no producer and isn't a fixed input, infer the producer
                if obj.config and obj.config.parameters["year"] == self.start_year:
                    print(f"Data {obj.name} is a fixed input for year {self.start_year}.")
                    return
                print(f"Data {obj.name} has no producer. Inferring producer...")
                producer = self.infer_producer(obj)
                obj.producer = producer
                self.build_dependencies(producer)

        elif isinstance(obj, ModelRun):
            # If the ModelRun is already registered, stop recursion
            if obj.name in self.model_registry:
                print(f"ModelRun {obj.name} is already registered.")
                return

            print(f"Resolving dependencies for model run: {obj.name}")
            # Resolve dependencies for each input
            for input_name, input_data in obj.inputs.items():
                if input_data:
                    self.build_dependencies(input_data)
            # Register the ModelRun
            print(f"Registering model run: {obj.name}")
            self.register_model_run(obj)
        else:
            raise ValueError(f"Unknown object type: {obj}")

    def resolve_workflow(self, final_model: ModelRun):
        """Resolves the entire workflow starting from the final model."""
        print(f"Resolving workflow starting from: {final_model.name}")
        self.build_dependencies(final_model)

    def execute_workflow(self):
        """Executes all registered model runs in dependency order."""
        print("--- Executing Workflow ---")
        for model_key, model in sorted(self.model_registry.items()):
            print(f"Running model: {model_key}")
            model.run()
        print("--- Workflow Execution Complete ---")


class Configuration:
    """Configuration for a specific model run."""

    def __init__(self, parameters: Dict[str, Any], state: GlobalRunState):
        self.parameters = parameters
        self.state = state

    def __repr__(self):
        return f"Configuration({self.parameters})"


class Plotter:
    """Class for visualizing a workflow managed by a GlobalConfiguration."""

    def __init__(self, global_config: GlobalConfiguration):
        self.global_config = global_config

    def generate_graph(self):
        """Generates a directed graph from the results of a completed run."""
        graph = nx.DiGraph()

        # Add nodes and edges for ModelRun and Data
        for model_key, model in self.global_config.model_registry.items():
            # Add the ModelRun node
            # name = "\n".join(model_key.split('_',1))
            name = model_key
            graph.add_node(name, label=model.name, type="ModelRun")

            # Add edges from input Data to ModelRun
            for input_name, input_data in model.inputs.items():
                if input_data:
                    graph.add_node(input_data.name, label=input_data.name, type="Data")
                    graph.add_edge(input_data.name, model_key, label=f"Input: {input_name}")

            # Add edges from ModelRun to output Data
            for output_data in model.outputs:
                if output_data:
                    graph.add_node(output_data.name, label=output_data.name, type="Data")
                    graph.add_edge(model_key, output_data.name, label="Output")

        # Ensure fixed inputs are added as standalone nodes
        for data_key, data in self.global_config.data_registry.items():
            if data.producer is None:  # Fixed inputs without a producer
                graph.add_node(data_key, label=data.name, type="Data")

        return graph

    def custom_layout(self, x_step=1.0, y_spacing=1.0):
        """
        Custom layout for the workflow graph.
        INPUT nodes are placed in a column corresponding to start_year - 1.
        Nodes are positioned by year and iteration on the x-axis and by type on the y-axis.
        """
        layout = {}
        start_year = self.global_config.start_year  # Access start_year from GlobalConfiguration
        input_column = start_year - 1  # Position INPUT nodes to the left of the first year
        iteration_step = 1.0 / self.global_config.iteration_settings.get("iterations_per_year", 1)

        i = -1
        # Position fixed input Data nodes
        for data_key, data in self.global_config.data_registry.items():
            if data.producer is None:  # Fixed input
                layout[data_key] = (input_column * x_step, i * y_spacing)  # INPUT column at x = start_year - 1
                i += 1

        # Position ModelRun and Data nodes
        for model_key, model in self.global_config.model_registry.items():
            year = model.config.parameters["year"]
            iteration = model.config.parameters.get("iteration", 0)
            x_position = year + (
                    (iteration or 0) * iteration_step)  # Increment x by 0.1 for each iteration within a year

            if "DemographicEvolution" in model.name:
                layout[model_key] = (x_position * x_step - 0.1, 1.5 * y_spacing)
            elif "ActivityDemand" in model.name:
                layout[model_key] = (x_position * x_step - 0.1, 0.5 * y_spacing)
            elif "NetworkSimulation" in model.name:
                layout[model_key] = (x_position * x_step - 0.1, -0.5 * y_spacing)
            else:
                layout[model_key] = (x_position * x_step + 0.1, -1.5 * y_spacing)

            # Position Data nodes produced by the ModelRun
            for output_data in model.outputs:
                if "ActivityPlans" in output_data.name:
                    layout[output_data.name] = (x_position * x_step, 0)
                elif "Population" in output_data.name:
                    layout[output_data.name] = (x_position * x_step, y_spacing)
                elif "Skims" in output_data.name:
                    layout[output_data.name] = (x_position * x_step, -y_spacing)  # Skims below Population

        # Add default positions for any nodes not explicitly handled
        graph = self.generate_graph()
        i = -1
        for node in graph.nodes:
            if node not in layout:
                layout[node] = (start_year - 2, i * y_spacing)  # Default position
                i += 1

        return layout

    def plot_graph(self, graph=None, x_step=1.0, y_spacing=1.0):
        """Plots the directed graph of the workflow with a custom layout."""
        if graph is None:
            graph = self.generate_graph()

        # Generate the custom layout
        pos = self.custom_layout(x_step=x_step, y_spacing=y_spacing)

        # Plot the graph
        plt.figure(figsize=(16, 10))
        node_colors = ["lightblue" if graph.nodes[node]["type"] == "ModelRun" else "lightgreen" for node in graph]
        nx.draw(
            graph, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=8
        )
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels={(u, v): d["label"] for u, v, d in graph.edges(data=True)}, font_size=8
        )
        plt.title("Workflow Visualization with INPUTs and Iterations")
        plt.show()


def custom_config_generator(state: GlobalRunState, iteration_settings: Dict[str, Any]) -> Dict[
    str, Any]:
    """Defines logic for creating configurations without database interaction."""
    base_config = {
        "model_name": state.model_name.split("_")[0],  # Use only the base name
        "year": state.year,
        "iteration": state.iteration,
    }

    # Example logic for configurations
    if "NetworkSimulation" in state.model_name:
        if state.year < 2025:
            base_config["param_x"] = "constant_value"
        else:
            base_config["param_x"] = "changed_value"

    return base_config


if __name__ == "__main__":
    # Create a GlobalConfiguration instance (assuming it's already set up)
    iteration_settings = {
        "workflow": ["DemographicEvolution", "NetworkSimulation"],
        "iterations_per_year": 1,
        "fixed_inputs": {
            "population": Data("Population", config_or_state=Configuration({"year": 2020}, state=GlobalRunState(
                "DemographicEvolution",
                2020, None)), producer=None),
            "skims": Data("Skims", config_or_state=Configuration({"year": 2020, "iteration": 0},
                                                        state=GlobalRunState(
                                                            "NetworkSimulation", 2020,
                                                            None)), producer=None),
        },
    }

    global_config = GlobalConfiguration(
        database_path="../scratch/test_workflow.db",
        start_year=2020,
        end_year=2021,
        iteration_settings=iteration_settings,
        custom_config_generator=custom_config_generator
    )

    final_outputs = global_config.run_workflow()

    # Create the Plotter instance
    plotter = Plotter(global_config)

    # Generate and plot the graph
    workflow_graph = plotter.generate_graph()
    plotter.plot_graph(workflow_graph, x_step=1.0, y_spacing=1.0)

    # Define the final Data object we want to generate
    final_skims = Skims(config=Configuration({"year": 2022, "iteration": 3}), producer=None)

    # Resolve dependencies starting from the final Data object
    print("--- Resolving Workflow Dependencies ---")
    global_config.build_dependencies(final_skims)

    global_config_iter = GlobalConfiguration(
        start_year=2020,
        end_year=2020,
        scenario="Baseline",
        model_settings={},
        iteration_settings={
            "fixed_inputs": {
                "population": Population(config=Configuration({"year": 2020})),  # Configuration specifies the year
                "skims": Skims(producer=None, config=Configuration({"year": 2020, "iteration": 0})),
            },
            "iterations_per_year": 2,
            "workflow": ["ActivityDemand", "NetworkSimulation"],
            "year_increment_after_loops": 1,
        }
    )

    # Lookup specific Data or ModelRun objects
    final_outputs = global_config_iter.run_iterative_workflow()

    # Print final outputs
    print("Final Activity Plans:", final_outputs.get("activity_plans"))
    print("Final Skims:", final_outputs.get("skims"))

    global_config_full = GlobalConfiguration(
        start_year=2020,
        end_year=2022,
        scenario="Baseline",
        model_settings={
            "DemographicEvolution": {"year_increment": 1},
            "ActivityDemand": {},
            "NetworkSimulation": {},
        },
        iteration_settings={
            "fixed_inputs": {
                "population": Population(config=Configuration({"year": 2020})),  # Configuration specifies the year
                "skims": Skims(producer=None, config=Configuration({"year": 2020, "iteration": 0})),
            },
            "iterations_per_year": 4,
        }
    )

    final_outputs_full = global_config_full.run_full_workflow()

    # Print final outputs
    print("Final Population:", final_outputs_full["final_population"])
    print("Final Activity Plans:", final_outputs_full["final_activity_plans"])
    print("Final Skims:", final_outputs_full["final_skims"])

    print("DONE")
