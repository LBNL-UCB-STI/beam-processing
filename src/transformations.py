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
        return events

    def updateMode(events):
        events["modeBEAM_rh"] = np.where(
            events.driver.str.contains("rideHailAgent", na=False),
            "ride_hail",
            events["modeBEAM"],
        )
        events["modeBEAM_rh"] = np.where(
            (events["type"] == "PathTraversal")
            & (events["modeBEAM"] == "car")
            & (events["driver"].str.contains("rideHailAgent", na=False))
            & (events["modeBEAM_rh_pooled"] != "nan"),
            events["modeBEAM_rh_pooled"],
            events["modeBEAM_rh"],
        )
        # We don't know if ridehail_transit is ride_hail or ride_hail_pooled
        events["modeBEAM_rh"] = np.where(
            (events["type"] == "PathTraversal")
            & (events["modeBEAM"] == "car")
            & (events["driver"].str.contains("rideHailAgent", na=False))
            & (events["modeBEAM_rh_ride_hail_transit"] != "nan"),
            events["modeBEAM_rh_ride_hail_transit"],
            events["modeBEAM_rh"],
        )

        # Dropping the temporary columns
        events = events.drop(["modeBEAM_rh_pooled"], axis=1)
        events = events.drop(["modeBEAM_rh_ride_hail_transit"], axis=1)
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
        events = events.rename(columns={"mode": "modeBEAM"})
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
        events = events.rename(columns={"mode": "modeBEAM", "person": "IDMerged"})
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

    def processPEVs(events):
        return 1

    def processModeChoice(events):
        events = events.rename(columns={"mode": "modeBEAM", "person": "IDMerged"})
        events["distance_travelling"] = np.where(
            (events["modeBEAM"] == "hov2_teleportation")
            | (events["modeBEAM"] == "hov3_teleportation"),
            events["length"],
            np.nan,
        )
        events["distance_mode_choice"] = np.where(
            events["type"] == "ModeChoice", events["length"], np.nan
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

    # def aggregator(grp):
    #     if grp.name in MCtimes.index:
    #         idx = (
    #             np.searchsorted(
    #                 MCtimes.loc[grp.name].values, grp["time"].values, side="right"
    #             )
    #             - 1
    #         )
    #         # vals = MCtimes.loc[grp.name].index.iloc[idx]
    #         grp["tripId"] = MCids.loc[grp.name].iloc[idx].values
    #     return grp["tripId"]
    #
    # PTs["tripId"] = PTs.groupby("IDMerged").apply(aggregator).explode().values

    dfs["PathTraversal"] = PTs
    dfs["TeleportationEvent"] = TEs
    dfs["ModeChoice"] = MCs

    return dfs


def assignTripIdToPathTraversals(pathTraversals, modeChoices):
    modeChoices.index = modeChoices.index.set_levels(
        modeChoices.index.levels[0].astype(int), level=0
    )
    MCtimes = modeChoices["original_time"].copy()
    MCids = modeChoices["tripId"].copy().astype(pd.Int64Dtype())

    def aggregator(grp):
        if grp.name in MCtimes.index:
            idx = np.searchsorted(
                MCtimes.loc[grp.name].values, grp["time"].values, side="right"
            )
            vals = MCids.loc[grp.name].iloc[idx - 1].values
        else:
            vals = np.full_like(grp.index, np.nan)
        return pd.Series(vals, index=grp.index.get_level_values(1))

    pathTraversals["tripId"] = pathTraversals.groupby("IDMerged").apply(aggregator)

    return pathTraversals


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
