import pandas as pd
import numpy as np
import geopandas as gpd


def getLinkStats(PTs: pd.DataFrame):
    """
    Calculates a replacement linkStats file based on the link travel times reported in path traversals

    Parameters:
        PTs (pd.DataFrame): Raw path traversal events

    Returns:
        pd.DataFrame: a dataframe of link volumes and travel times from the path traversal events
    """
    PTs = PTs.head(10000)
    linksAndTravelTimes = pd.concat(
        [
            PTs.links.str.split(","),
            PTs.linkTravelTime.str.split(","),
            PTs.departureTime,
        ],
        axis=1,
    ).explode(["links", "linkTravelTime"])
    linksAndTravelTimes["linkTravelTime"] = linksAndTravelTimes[
        "linkTravelTime"
    ].astype(float)
    linksAndTravelTimes["links"] = linksAndTravelTimes["links"].astype(pd.Int64Dtype())
    linksAndTravelTimes = linksAndTravelTimes.loc[
        linksAndTravelTimes.index.duplicated(keep="first")
    ]
    linksAndTravelTimes["cumulativeTravelTime"] = linksAndTravelTimes.groupby(
        level=0
    ).agg({"linkTravelTime": np.cumsum})
    linksAndTravelTimes["hour"] = np.floor(
        (
            linksAndTravelTimes["cumulativeTravelTime"]
            + linksAndTravelTimes["departureTime"]
        )
        / 3600.0
    )
    linksAndTravelTimes["volume"] = 1.0
    grouped = linksAndTravelTimes.groupby(["links", "hour"]).agg(
        {"linkTravelTime": np.sum, "volume": np.sum}
    )
    print("Aggregating links into size {0}".format(grouped.index.levshape))
    grouped.index.set_names(["link", "hour"], inplace=True)
    return grouped.rename(columns={"linkTravelTime": "traveltime"})


def fixPathTraversals(PTs: pd.DataFrame):
    """
    Adds some additional columns to a dataframe of path traversal events, including
    corrected occupancy, vehicle miles, passenger miles, and a mode_extended column
    that differentiates ridehail

    Parameters:
        PTs (pd.DataFrame): Raw path traversal events

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    PTs["duration"] = PTs["arrivalTime"] - PTs["departureTime"]
    PTs["mode_extended"] = PTs["mode"]
    PTs["isRH"] = PTs["vehicle"].str.contains("rideHail")
    PTs["isCAV"] = PTs["vehicleType"].str.contains("L5")
    PTs.loc[PTs["isRH"], "mode_extended"] += "_RideHail"
    PTs.loc[PTs["isCAV"], "mode_extended"] += "_CAV"
    PTs["occupancy"] = PTs["numPassengers"]
    PTs.loc[PTs["mode_extended"] == "car", "occupancy"] += 1
    PTs.loc[PTs["mode_extended"] == "walk", "occupancy"] = 1
    PTs.loc[PTs["mode_extended"] == "bike", "occupancy"] = 1
    PTs["vehicleMiles"] = PTs["length"] / 1609.34
    PTs["passengerMiles"] = (PTs["length"] * PTs["occupancy"]) / 1609.34
    PTs["totalEnergyInJoules"] = PTs["primaryFuel"] + PTs["secondaryFuel"]
    PTs["gallonsGasoline"] = 0.0
    PTs.loc[PTs["primaryFuelType"] == "gasoline", "gallonsGasoline"] += (
        PTs.loc[PTs["primaryFuelType"] == "gasoline", "primaryFuel"] * 8.3141841e-9
    )
    PTs.loc[PTs["secondaryFuelType"] == "gasoline", "gallonsGasoline"] += (
        PTs.loc[PTs["secondaryFuelType"] == "gasoline", "secondaryFuel"] * 8.3141841e-9
    )
    PTs.drop(
        columns=[
            "numPassengers",
            "length",
            "type",
            "primaryFuelLevel",
            "secondaryFuelLevel",
            "fromStopIndex",
            "toStopIndex",
            "capacity",
            "seatingCapacity",
            "toStopIndex",
        ],
        inplace=True,
    )
    return PTs.convert_dtypes()


def filterPersons(persons: pd.DataFrame):
    return persons.loc[
        :,
        [
            "earning",
            "worker",
            "student",
            "household_id",
            "school_zone_id",
            "age",
            "work_zone_id",
            "TAZ",
            "home_x",
            "home_y",
        ],
    ].copy()


def filterHouseholds(households: pd.DataFrame):
    return households.loc[
        :,
        [
            "recent_mover",
            "num_workers",
            "sf_detached",
            "tenure",
            "race_of_head",
            "income",
            "block_id",
            "cars",
            "hhsize",
            "TAZ",
            "num_drivers",
            "num_children",
        ],
    ].copy()


def filterTrips(trips: pd.DataFrame):
    return trips.loc[
        :,
        [
            "person_id",
            "household_id",
            "tour_id",
            "primary_purpose",
            "purpose",
            "destination",
            "origin",
            "destination_logsum",
            "depart",
            "trip_mode",
            "mode_choice_logsum",
        ],
    ].copy()


def doInexus(dfs: dict):
    def addEmissions(events):
        events = events.copy()
        events["fuelFood"] = np.where(
            (events["type"] == "PathTraversal") & (events["primaryFuelType"] == "Food"),
            events["primaryFuel"],
            np.nan,
        )
        events["emissionFood"] = events["fuelFood"] * 8.3141841e-9 * 0
        events["fuelElectricity"] = np.where(
            (events["type"] == "PathTraversal")
            & (events["primaryFuelType"] == "Electricity"),
            events["primaryFuel"],
            np.nan,
        )
        events["emissionElectricity"] = (
            events["fuelElectricity"] * 2.77778e-10 * 947.2 * 0.0005
        )
        events["fuelDiesel"] = np.where(
            (events["type"] == "PathTraversal")
            & (events["primaryFuelType"] == "Diesel"),
            events["primaryFuel"],
            np.nan,
        )
        events["emissionDiesel"] = events["fuelDiesel"] * 8.3141841e-9 * 10.180e-3
        events["fuelBiodiesel"] = np.where(
            (events["type"] == "PathTraversal")
            & (events["primaryFuelType"] == "Biodiesel"),
            events["primaryFuel"],
            np.nan,
        )
        events["emissionBiodiesel"] = events["fuelBiodiesel"] * 8.3141841e-9 * 10.180e-3
        events["fuel_not_Food"] = np.where(
            (events["type"] == "PathTraversal") & (events["primaryFuelType"] != "Food"),
            events["primaryFuel"] + events["secondaryFuel"],
            np.nan,
        )
        events["fuelGasoline"] = np.where(
            (events["type"] == "PathTraversal")
            & (
                (events["primaryFuelType"] == "Gasoline")
                | (events["secondaryFuelType"] == "Gasoline")
            ),
            events["primaryFuel"] + events["secondaryFuel"],
            np.nan,
        )
        events["emissionGasoline"] = events["fuelGasoline"] * 8.3141841e-9 * 8.89e-3

        # Marginal fuel
        conditions = [
            (events["modeBEAM"] == "ride_hail_pooled"),
            (events["modeBEAM"] == "walk_transit")
            | (events["modeBEAM"] == "drive_transit")
            | (events["modeBEAM"] == "ride_hail_transit")
            | (events["modeBEAM"] == "bus")
            | (events["modeBEAM"] == "subway")
            | (events["modeBEAM"] == "rail")
            | (events["modeBEAM"] == "tram")
            | (events["modeBEAM"] == "cable_car")
            | (events["modeBEAM"] == "bike_transit"),
            (events["modeBEAM"] == "walk") | (events["modeBEAM"] == "bike"),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation"),
        ]
        choices = [
            events["fuel_not_Food"] / events["numPassengers"],
            0,
            events["fuelFood"],
            events["fuel_not_Food"],
        ]
        events["fuel_marginal"] = np.select(conditions, choices, default=np.nan)

        # Marginal emission
        conditions1 = [
            (events["modeBEAM"] == "ride_hail_pooled")
            & (events["fuelElectricity"].notna() != 0),
            (events["modeBEAM"] == "ride_hail_pooled")
            & (events["fuelGasoline"].notna() != 0),
            (events["modeBEAM"] == "ride_hail_pooled")
            & (events["fuelBiodiesel"].notna() != 0),
            (events["modeBEAM"] == "ride_hail_pooled")
            & (events["fuelDiesel"].notna() != 0),
            (events["modeBEAM"] == "walk_transit")
            | (events["modeBEAM"] == "drive_transit")
            | (events["modeBEAM"] == "ride_hail_transit")
            | (events["modeBEAM"] == "bus")
            | (events["modeBEAM"] == "subway")
            | (events["modeBEAM"] == "rail")
            | (events["modeBEAM"] == "tram")
            | (events["modeBEAM"] == "cable_car")
            | (events["modeBEAM"] == "bike_transit"),
            (events["modeBEAM"] == "walk") | (events["modeBEAM"] == "bike"),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation")
            & (events["fuelElectricity"].notna() != 0),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation")
            & (events["fuelGasoline"].notna() != 0),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation")
            & (events["fuelBiodiesel"].notna() != 0),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation")
            & (events["fuelDiesel"].notna() != 0),
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation")
            & (events["fuelFood"].notna() != 0),
        ]

        choices1 = [
            events["emissionElectricity"] / events["numPassengers"],
            events["emissionGasoline"] / events["numPassengers"],
            events["emissionBiodiesel"] / events["numPassengers"],
            events["emissionDiesel"] / events["numPassengers"],
            0,
            events["emissionFood"],
            events["emissionElectricity"],
            events["emissionGasoline"],
            events["emissionBiodiesel"],
            events["emissionDiesel"],
            events["emissionFood"],
        ]

        events["emission_marginal"] = np.select(conditions1, choices1, default=np.nan)

        return events

    def updateDuration(events):
        events = events.copy()
        events["duration_travelling"] = events["arrivalTime"] - events["departureTime"]
        events["distance_travelling"] = events["length"].copy()
        events["duration_walking"] = np.where(
            events["modeBEAM"] == "walk", events["duration_travelling"], np.nan
        )
        events["distance_walking"] = np.where(
            events["modeBEAM"] == "walk", events["distance_travelling"], np.nan
        )
        events["duration_on_bike"] = np.where(
            events["modeBEAM"] == "bike", events["duration_travelling"], np.nan
        )
        events["distance_bike"] = np.where(
            events["modeBEAM"] == "bike", events["distance_travelling"], np.nan
        )
        events["duration_in_ridehail"] = np.where(
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "ride_hail_pooled")
            | (events["modeBEAM"] == "ride_hail_transit"),
            events["duration_travelling"],
            np.nan,
        )
        events["distance_ridehail"] = np.where(
            (events["modeBEAM"] == "ride_hail")
            | (events["modeBEAM"] == "ride_hail_pooled")
            | (events["modeBEAM"] == "ride_hail_transit"),
            events["distance_travelling"],
            np.nan,
        )
        events["duration_in_privateCar"] = np.where(
            (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation"),
            events["duration_travelling"],
            np.nan,
        )
        events["distance_privateCar"] = np.where(
            (events["modeBEAM"] == "car")
            | (events["modeBEAM"] == "car_hov3")
            | (events["modeBEAM"] == "car_hov2")
            | (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation"),
            events["distance_travelling"],
            np.nan,
        )
        events["duration_in_transit"] = np.where(
            (events["modeBEAM"] == "bike_transit")
            | (events["modeBEAM"] == "drive_transit")
            | (events["modeBEAM"] == "walk_transit")
            | (events["modeBEAM"] == "bus")
            | (events["modeBEAM"] == "tram")
            | (events["modeBEAM"] == "subway")
            | (events["modeBEAM"] == "rail")
            | (events["modeBEAM"] == "cable_car")
            | (events["modeBEAM"] == "ride_hail_transit"),
            events["duration_travelling"],
            np.nan,
        )
        events["distance_transit"] = np.where(
            (events["modeBEAM"] == "bike_transit")
            | (events["modeBEAM"] == "drive_transit")
            | (events["modeBEAM"] == "walk_transit")
            | (events["modeBEAM"] == "bus")
            | (events["modeBEAM"] == "tram")
            | (events["modeBEAM"] == "subway")
            | (events["modeBEAM"] == "rail")
            | (events["modeBEAM"] == "cable_car")
            | (events["modeBEAM"] == "ride_hail_transit"),
            events["distance_travelling"],
            np.nan,
        )
        return events

    def processPTs(events):
        events = events.rename(columns={"mode": "modeBEAM", "netCost": "cost_BEAM"})
        events = addEmissions(events)
        # events = updateMode(events)
        events = updateDuration(events)

        isPublicVehicleTraversal = events.driver.str.contains("Agent")
        privateVehicleEvents = events.loc[~isPublicVehicleTraversal, :].copy()
        privateVehicleEvents["IDMerged"] = privateVehicleEvents["driver"].copy()
        privateVehicleEvents["IDMerged"] = pd.to_numeric(privateVehicleEvents.IDMerged)

        publicVehicleEvents = events.loc[isPublicVehicleTraversal, :]
        publicVehicleEvents = publicVehicleEvents.loc[
            ~publicVehicleEvents.riders.isna(), :
        ].copy()
        publicVehicleEvents["riderList"] = publicVehicleEvents["riders"].str.split(":")
        publicVehicleEvents = publicVehicleEvents.explode("riderList")
        publicVehicleEvents["IDMerged"] = publicVehicleEvents["riderList"].copy()
        publicVehicleEvents.drop(columns=["riderList"], inplace=True)
        publicVehicleEvents["IDMerged"] = pd.to_numeric(publicVehicleEvents.IDMerged)
        publicVehicleEvents["transit_bus"] = np.where(
            publicVehicleEvents["modeBEAM"] == "bus", 1, 0
        )
        publicVehicleEvents["transit_subway"] = np.where(
            publicVehicleEvents["modeBEAM"] == "subway", 1, 0
        )
        publicVehicleEvents["transit_tram"] = np.where(
            publicVehicleEvents["modeBEAM"] == "tram", 1, 0
        )
        publicVehicleEvents["transit_rail"] = np.where(
            publicVehicleEvents["modeBEAM"] == "rail", 1, 0
        )
        publicVehicleEvents["transit_cable_car"] = np.where(
            publicVehicleEvents["modeBEAM"] == "cable_car", 1, 0
        )

        pathTraversals = (
            pd.concat([publicVehicleEvents, privateVehicleEvents], axis=0)
            .sort_values(["time"])
            .reset_index(drop=True)
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )

        pathTraversals["eventOrder"] = (
            pathTraversals.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        pathTraversals = pathTraversals.set_index("eventOrder", append=True).droplevel(
            1
        )
        pathTraversals.drop(
            columns=[
                "driver",
                "riders",
                "toStopIndex",
                "fromStopIndex",
                "seatingCapacity",
                "linkTravelTime",
                "secondaryFuel",
                "secondaryFuelType",
                "primaryFuelType",
                "links",
                "primaryFuel",
                "secondaryFuelLevel",
                "primaryFuelLevel",
                "currentTourMode",
            ],
            inplace=True,
        )
        return pathTraversals

    def processTeleportation(events):
        events = events.copy().rename(
            columns={"currentTourMode": "mode_choice_actual_BEAM", "person": "IDMerged"}
        )

        events["duration_travelling"] = events["arrivalTime"] - events["departureTime"]

        events["duration_in_privateCar"] = events["duration_travelling"].copy()

        events["mode_choice_planned_BEAM"] = events["mode_choice_actual_BEAM"].copy()

        events["IDMerged"] = pd.to_numeric(events.IDMerged)
        events = (
            events.sort_values(["time"])
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )

        events["eventOrder"] = (
            events.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        return events.set_index("eventOrder", append=True).droplevel(1)

    def processReplanning(events):
        # TODO: Check that this gets indexed correctly
        events = events.copy().rename(columns={"person": "IDMerged"})
        events["IDMerged"] = pd.to_numeric(events.IDMerged)
        events = (
            events.sort_values(["time"])
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )
        events["replanning_status"] = 1

        events["eventOrder"] = (
            events.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        return events.set_index("eventOrder", append=True).droplevel(1)

    def processParking(events):
        events = events.rename(columns={"cost": "cost_BEAM", "driver": "IDMerged"})
        events = events.loc[events["IDMerged"].str.isnumeric(), :]
        events = events.loc[
            ~events["IDMerged"].isna(),  # Might be duplicated
            ["IDMerged", "parkingTaz", "parkingType", "time", "type", "cost_BEAM"],
        ].copy()
        events = (
            events.sort_values(["time"])
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )

        events["eventOrder"] = (
            events.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        return events.set_index("eventOrder", append=True).droplevel(1)

    def processPersonCost(events):
        events = events.rename(columns={"person": "IDMerged", "mode": "mode_BEAM"})
        events["cost_BEAM"] = events["tollCost"] + events["netCost"]
        events = events[["IDMerged", "mode_BEAM", "time", "type", "cost_BEAM"]].copy()
        events = (
            events.sort_values(["time"])
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )

        events["eventOrder"] = (
            events.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        return events.set_index("eventOrder", append=True).droplevel(1)

    def processModeChoice(events):
        events = events.rename(
            columns={
                "mode": "modeBEAM",
                "person": "IDMerged",
                "netCost": "cost_BEAM",
                "length": "distance_mode_choice",
            }
        )
        events["mode_choice_actual_BEAM"] = events.groupby(["IDMerged", "tripId"])[
            "modeBEAM"
        ].transform("last")
        events["mode_choice_planned_BEAM"] = events.groupby(["IDMerged", "tripId"])[
            "modeBEAM"
        ].transform("first")
        events["original_time"] = events.groupby(["IDMerged", "tripId"])[
            "time"
        ].transform("first")

        events.drop_duplicates(subset=["tripId"], keep="last", inplace=True)

        events = events.drop(
            columns=[
                "availableAlternatives",
                "tourIndex",
                "legModes",
                "legVehicleIds",
                "personalVehicleAvailable",
                "currentTourMode",
            ]
        )
        events = (
            events.sort_values(["time"])
            .set_index("IDMerged", append=True)
            .reorder_levels([1, 0])
            .sort_index(level=0)
        )

        events["eventOrder"] = (
            events.index.to_frame(index=False)
            .groupby("IDMerged")
            .agg("rank")
            .astype(int)
            .values
        )
        return events.set_index("eventOrder", append=True).droplevel(1)

    PTs = processPTs(dfs["PathTraversal"])
    TEs = processTeleportation(dfs["TeleportationEvent"])
    MCs = processModeChoice(dfs["ModeChoice"])
    REs = processReplanning(dfs["Replanning"])
    PEs = processParking(dfs["ParkingEvent"])
    PCs = processPersonCost(dfs["PersonCost"])

    dfs["PathTraversal"] = PTs
    dfs["TeleportationEvent"] = TEs
    dfs["ModeChoice"] = MCs
    dfs["Replanning"] = REs
    dfs["ParkingEvent"] = PEs
    dfs["PersonCost"] = PCs

    return dfs


def assignTripIdToEvents(pathTraversals, modeChoices, otherColumns=None):
    if otherColumns is None:
        otherColumns = dict()
    modeChoices.index = modeChoices.index.set_levels(
        modeChoices.index.levels[0].astype(int), level=0
    )
    MCtimes = modeChoices["original_time"].copy()
    MCids = modeChoices["tripId"].copy().astype(pd.Int64Dtype())
    # toAdd = dict()
    # for col in otherColumns:
    #     toAdd[col] = modeChoices[col].copy()

    def aggregator(grp):
        vals = dict()
        pId = int(grp.name)
        if pId in MCtimes.index:
            idx = np.searchsorted(
                MCtimes.loc[pId].values, grp["time"].values, side="right"
            )
            vals["tripId"] = MCids.loc[pId].iloc[idx - 1].values
            for oldName, newName in otherColumns.items():
                vals[newName] = modeChoices.loc[pId, oldName].iloc[idx - 1].values
        return pd.DataFrame(vals, index=grp.index.get_level_values(1))

    newColumns = pathTraversals.groupby("IDMerged").apply(aggregator)
    return pd.concat([pathTraversals, newColumns], axis=1)


def mergeWithTripsAndAggregate(events, trips, utilities, persons):
    # aggfunc = {'actStartTime': "sum",
    #            'actEndTime': "sum",
    aggfunc = {
        "duration_travelling": "sum",
        "cost_BEAM": "sum",
        # 'actStartType': "sum",
        # 'actEndType': "sum",
        "duration_walking": "sum",
        "duration_in_privateCar": "sum",
        "duration_on_bike": "sum",
        "duration_in_ridehail": "sum",
        "distance_travelling": "sum",
        "duration_in_transit": "sum",
        "distance_walking": "sum",
        "distance_bike": "sum",
        "distance_ridehail": "sum",
        "distance_privateCar": "sum",
        "distance_transit": "sum",
        # 'legVehicleIds': "sum",
        "mode_choice_planned_BEAM": "first",
        "mode_choice_actual_BEAM": "last",
        "vehicle": lambda x: ", ".join(set(x.dropna().astype(str))),
        "numPassengers": lambda x: ", ".join(list(x.dropna().astype(str))),
        "distance_mode_choice": "sum",
        "replanning_status": "sum",
        "reason": lambda x: ", ".join(list(x.dropna().astype(str))),
        "parkingType": lambda x: ", ".join(list(x.dropna().astype(str))),
        "transit_bus": "sum",
        "transit_subway": "sum",
        "transit_tram": "sum",
        "transit_cable_car": "sum",
        # 'ride_hail_pooled': "sum",
        "transit_rail": "sum",
        "fuelFood": "sum",
        "fuelElectricity": "sum",
        "fuelBiodiesel": "sum",
        "fuelDiesel": "sum",
        "fuel_not_Food": "sum",
        "fuelGasoline": "sum",
        'fuel_marginal': "sum",
        # 'BlockGroupStart': 'first',
        "startX": "first",
        "startY": "first",
        # 'bgid_start': 'first',
        # 'tractid_start': 'first',
        # 'juris_name_start': 'first',
        # 'county_name_start': 'first',
        # 'mpo_start': 'first',
        # 'BlockGroupEnd': 'last',
        "endX": "last",
        "endY": "last",
        # 'bgid_end': 'last',
        # 'tractid_end': 'last',
        # 'juris_name_end': 'last',
        # 'county_name_end': 'last',
        # 'mpo_end': 'last',
        "emissionFood": "sum",
        "emissionElectricity": "sum",
        "emissionDiesel": "sum",
        "emissionGasoline": "sum",
        "emissionBiodiesel": "sum",
        'emission_marginal': "sum"
    }
    p_cols = [
        "age",
        "earning",
        "edu",
        "race_id",
        "sex",
        "household_id",
        "home_taz",
        "school_taz",
        "workplace_taz",
        "workplace_location_logsum",
        "distance_to_work",
    ]

    t_cols = [
        "person_id",
        "tour_id",
        "primary_purpose",
        "purpose",
        "destination",
        "origin",
        "destination_logsum",
        "depart",
        "trip_mode",
        "mode_choice_logsum",
    ]

    eventsByTrip = events.groupby("tripId").agg(aggfunc)

    asimData = pd.merge(
        pd.merge(utilities, trips[t_cols], left_on="trip_id", right_index=True),
        persons[p_cols],
        left_on="person_id",
        right_index=True,
    )

    final = pd.merge(
        eventsByTrip.reset_index(),
        asimData,
        left_on="tripId",
        right_on="trip_id",
        how="outer",
    )
    return final


def labelNetworkWithTaz(network: pd.DataFrame, TAZ: gpd.GeoDataFrame):
    gdf = gpd.GeoDataFrame(
        network,
        geometry=gpd.points_from_xy(
            network["toLocationX"], network["toLocationY"], crs="epsg:26910"
        ),
    ).sjoin(TAZ.loc[:, ["taz1454", "geometry"]], how="left")
    return pd.DataFrame(gdf.drop(columns=["geometry", "index_right"]))


def mergeLinkstatsWithNetwork(linkStats: pd.DataFrame, network: pd.DataFrame):
    linkStats["VMT"] = linkStats["volume"] * linkStats["length"] / 1609.34
    linkStats["VHT"] = linkStats["volume"] * linkStats["traveltime"] / 3600.0
    out = linkStats.merge(network, left_on="link", right_index=True)
    return out[
        list(linkStats.columns)
        + [
            "linkLength",
            "linkFreeSpeed",
            "linkCapacity",
            "numberOfLanes",
            "linkModes",
            "attributeOrigId",
            "attributeOrigType",
            "taz1454",
        ]
    ]
