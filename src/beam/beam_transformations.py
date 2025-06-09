from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


def getLinkStatsFromPathTraversals(PTs: pd.DataFrame):
    """
    Calculates a replacement linkStats file based on the link travel times reported in path traversals

    Parameters:
        PTs (pd.DataFrame): Raw path traversal events

    Returns:
        pd.DataFrame: a dataframe of link volumes and travel times from the path traversal events
    """
    linksAndTravelTimes = pd.concat(
        [
            PTs.links.str.split(","),
            PTs.linkTravelTime.str.split(","),
            PTs.departureTime,
        ],
        axis=1,
    ).explode(["links", "linkTravelTime"])
    linksAndTravelTimes = linksAndTravelTimes.loc[
        linksAndTravelTimes["links"] != "", :
    ].copy()
    linksAndTravelTimes["linkTravelTime"] = pd.to_numeric(
        linksAndTravelTimes["linkTravelTime"]
    )
    linksAndTravelTimes["links"] = linksAndTravelTimes["links"].astype(pd.Int64Dtype())
    # linksAndTravelTimes = linksAndTravelTimes.loc[
    #     ~linksAndTravelTimes.index.duplicated(keep="first")
    # ]
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
    PTs.loc[PTs["mode_extended"] == "car_hov2", "occupancy"] = 2
    PTs.loc[PTs["mode_extended"] == "car_hov3", "occupancy"] = 3
    PTs.loc[
        (PTs["occupancy"] == 2) & (PTs["mode_extended"] == "car"), "mode_extended"
    ] = "car_hov2"
    PTs.loc[
        (PTs["occupancy"] == 3) & (PTs["mode_extended"] == "car"), "mode_extended"
    ] = "car_hov3"
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
    toDrop = [
            "type",
            "primaryFuelLevel",
            "secondaryFuelLevel",
            "fromStopIndex",
            "toStopIndex",
            "capacity",
            "seatingCapacity",
            "toStopIndex",
        ]
    PTs.drop(
        columns=[col for col in toDrop if col in PTs.columns],
        inplace=True,
    )
    return PTs.convert_dtypes()


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
        events = updateDuration(events)

        isPublicVehicleTraversal = events.driver.str.contains("Agent")
        privateVehicleEvents = events.loc[~isPublicVehicleTraversal, :].copy()
        privateVehicleEvents["IDMerged"] = privateVehicleEvents["driver"].copy()
        privateVehicleEvents["IDMerged"] = pd.to_numeric(privateVehicleEvents.IDMerged)

        publicVehicleEvents = events.loc[isPublicVehicleTraversal, :]
        # Only process public vehicle events with riders
        publicVehicleEvents = publicVehicleEvents.loc[
            ~publicVehicleEvents.riders.isna(), :
        ].copy()
        publicVehicleEvents["riderList"] = publicVehicleEvents["riders"].str.split(":")
        publicVehicleEvents = publicVehicleEvents.explode("riderList")
        publicVehicleEvents["IDMerged"] = publicVehicleEvents["riderList"].copy()
        publicVehicleEvents.drop(columns=["riderList"], inplace=True)
        publicVehicleEvents["IDMerged"] = pd.to_numeric(publicVehicleEvents.IDMerged)
        # Add specific transit mode flags
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

        # Combine public and private events, sort by time within each person
        pathTraversals = pd.concat(
            [publicVehicleEvents, privateVehicleEvents], axis=0
        ).sort_values(
            ["IDMerged", "time"]
        )  # Ensure sorting by person then time

        pathTraversals["eventOrder"] = pathTraversals.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        pathTraversals = pathTraversals.set_index(["IDMerged", "eventOrder"])

        # Drop columns that are no longer needed or processed
        pathTraversals.drop(
            columns=[
                "driver",
                "riders",
                "toStopIndex",
                "fromStopIndex",
                "seatingCapacity",
                "linkTravelTime",  # linkTravelTime was exploded and processed, raw column can be dropped
                "secondaryFuel",  # fuels are aggregated in addEmissions
                "secondaryFuelType",
                "primaryFuelType",
                "links",  # links was exploded and processed, raw column can be dropped
                "primaryFuel",  # fuels are aggregated in addEmissions
                "secondaryFuelLevel",
                "primaryFuelLevel",
                # "currentTourMode", # Keep if used later, remove if not
            ],
            inplace=True,
            errors="ignore",  # Ignore if column doesn't exist
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
        # Sort by IDMerged then time
        events = events.sort_values(["IDMerged", "time"])

        # Fix eventOrder calculation - use cumcount for sequence number per person
        events["eventOrder"] = events.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        return events.set_index(["IDMerged", "eventOrder"])

    def processReplanning(events):
        events = events.copy().rename(columns={"person": "IDMerged"})
        events["IDMerged"] = pd.to_numeric(events.IDMerged)
        # Sort by IDMerged then time
        events = events.sort_values(["IDMerged", "time"])
        events["replanning_status"] = 1

        # Fix eventOrder calculation - use cumcount for sequence number per person
        events["eventOrder"] = events.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        return events.set_index(["IDMerged", "eventOrder"])

    def processParking(events):
        events = events.rename(columns={"cost": "cost_BEAM", "driver": "IDMerged"})
        events = events.loc[
            ~events["IDMerged"].isna(),  # Might be duplicated
            ["IDMerged", "parkingTaz", "parkingType", "time", "type", "cost_BEAM"],
        ]
        # Ensure IDMerged is numeric after filtering
        events = events.loc[
            pd.to_numeric(events["IDMerged"], errors="coerce").notna(), :
        ].copy()
        events["IDMerged"] = pd.to_numeric(events.IDMerged)

        # Sort by IDMerged then time
        events = events.sort_values(["IDMerged", "time"])

        # Fix eventOrder calculation - use cumcount for sequence number per person
        events["eventOrder"] = events.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        return events.set_index(["IDMerged", "eventOrder"])

    def processPersonCost(events):
        events = events.rename(columns={"person": "IDMerged", "mode": "mode_BEAM"})
        events["cost_BEAM"] = events["tollCost"] + events["netCost"]
        events = events[["IDMerged", "mode_BEAM", "time", "type", "cost_BEAM"]].copy()
        # Ensure IDMerged is numeric
        events["IDMerged"] = pd.to_numeric(events.IDMerged)

        # Sort by IDMerged then time
        events = events.sort_values(["IDMerged", "time"])

        # Fix eventOrder calculation - use cumcount for sequence number per person
        events["eventOrder"] = events.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        return events.set_index(["IDMerged", "eventOrder"])

    def processModeChoice(events):
        events = events.rename(
            columns={
                "mode": "modeBEAM",
                "person": "IDMerged",
                "netCost": "cost_BEAM",
                "length": "distance_mode_choice",  # Note: length is path length, not crow-flies distance
            }
        )
        # Ensure IDMerged is numeric
        events["IDMerged"] = pd.to_numeric(events.IDMerged)

        # Sort by IDMerged then time before calculating actual/planned
        events = events.sort_values(["IDMerged", "time"])

        # Keep only the first (planned) and last (actual) mode choice per tripId
        events["mode_choice_actual_BEAM"] = events.groupby(["IDMerged", "tripId"])[
            "modeBEAM"
        ].transform("last")
        events["mode_choice_planned_BEAM"] = events.groupby(["IDMerged", "tripId"])[
            "modeBEAM"
        ].transform("first")
        events["original_time"] = events.groupby(["IDMerged", "tripId"])[
            "time"
        ].transform("first")

        # Drop duplicate trip IDs, keeping the last event for each trip
        events.drop_duplicates(subset=["IDMerged", "tripId"], keep="last", inplace=True)

        events = events.drop(
            columns=[
                "availableAlternatives",
                "tourIndex",
                "legModes",
                "legVehicleIds",
                "personalVehicleAvailable",
                "currentTourMode",  # Keep or remove as needed based on usage elsewhere
            ],
            errors="ignore",  # Ignore if column doesn't exist
        )

        # Sort by IDMerged then time for final indexing
        events = events.sort_values(["IDMerged", "time"])

        # Fix eventOrder calculation - use cumcount for sequence number per person
        events["eventOrder"] = events.groupby("IDMerged").cumcount() + 1

        # Set desired final index (IDMerged, eventOrder)
        # Note: With drop_duplicates above, this index might not be unique if multiple trips per person
        # but it seems the goal is to index events *by* person, so this might be okay.
        # If index should be unique per trip, tripId should be in index. Revisit if needed.
        return events.set_index(["IDMerged", "eventOrder"])

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
    # Assigns tripId and other columns from ModeChoice events to other event types (like PathTraversal)
    # by matching events occurring between the start and end times of a trip's mode choice sequence.
    # Assumes modeChoices and pathTraversals are indexed by (IDMerged, eventOrder) and sorted by time implicitly through eventOrder.

    if otherColumns is None:
        otherColumns = dict()

    # Extract relevant columns from ModeChoices for quick lookup
    # We need IDMerged, original_time (trip start time), and tripId
    # Plus any columns specified in otherColumns
    mc_lookup_cols = ["original_time", "tripId"] + list(otherColumns.keys())
    # Get mode choices data as a DataFrame with IDMerged as a regular column for groupby
    mc_df = modeChoices[mc_lookup_cols].reset_index(
        "eventOrder"
    )  # Drop eventOrder from index temporarily

    # Helper function to apply to each person's events
    def aggregator(grp):
        # grp is a DataFrame for a single IDMerged, indexed by eventOrder (and implicitly sorted by time)
        person_id = grp.name  # Get the person ID

        # Get mode choices for this person, sorted by time
        person_mc = mc_df.loc[person_id].sort_values("original_time")

        vals = {}
        if not person_mc.empty:
            # Get the times of mode choices (trip start times) for this person
            mc_times = person_mc["original_time"].values
            # Get the corresponding tripIds
            mc_trip_ids = person_mc["tripId"].values
            # Get values for other requested columns
            other_cols_data = {
                col: person_mc[col].values for col in otherColumns.keys()
            }

            # For each event in the current person's events (grp), find which trip's time range it falls into.
            # np.searchsorted finds the insertion point for each event time ('time' column in grp)
            # in the sorted list of mode choice times (mc_times).
            # side='right' means events happening *at* the start time are included in the previous trip.
            # We subtract 1 from the index because searchsorted gives the index *after* the match.
            idx = np.searchsorted(mc_times, grp["time"].values, side="right") - 1

            # Ensure indices are within bounds (handle events before the first trip or after the last)
            idx = np.clip(idx, 0, len(mc_times) - 1)

            # Assign the tripId and other column values corresponding to the found index
            vals["tripId"] = mc_trip_ids[idx]
            for oldName, newName in otherColumns.items():
                vals[newName] = other_cols_data[oldName][idx]

        # Return a DataFrame with the new columns, indexed the same way as the input grp (by eventOrder)
        if not vals:  # If no mode choices for this person
            # Return an empty DataFrame with the expected index and columns
            return pd.DataFrame(
                columns=["tripId"] + list(otherColumns.values()), index=grp.index
            )

        # Create DataFrame using grp.index which should be (IDMerged, eventOrder)
        result_df = pd.DataFrame(vals, index=grp.index)

        # Ensure column names for otherColumns are the 'newName'
        result_df.rename(columns=otherColumns, inplace=True)

        return result_df

    # Apply the aggregator to each person group
    newColumns = pathTraversals.groupby("IDMerged").apply(aggregator)

    # Concatenate the original pathTraversals with the new columns
    # The index of pathTraversals is (IDMerged, eventOrder).
    # The index of newColumns is (IDMerged, eventOrder) because we used grp.index in aggregator.
    # Direct concatenation should work.
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
        "fuel_marginal": "sum",
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
        "emission_marginal": "sum",
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


def labelNetworkWithTaz(
    network: pd.DataFrame,
    TAZ: gpd.GeoDataFrame,
    taz_column: str,
    crs: str = "epsg:26910",
):
    gdf = gpd.GeoDataFrame(
        network,
        geometry=gpd.points_from_xy(
            network["toLocationX"],
            network["toLocationY"],
            crs=crs,  # TODO: Match CRS with TAZ
        ),
    ).sjoin(TAZ.loc[:, [taz_column, "geometry"]], how="left")
    return pd.DataFrame(gdf.drop(columns=["geometry", "index_right"]))


def mergeLinkstatsWithNetwork(
    linkStats: pd.DataFrame, network: pd.DataFrame, index: Optional[str] = None
):
    out = network.merge(linkStats, right_on="link", left_index=True)
    out["VMT"] = out["volume"] * out["linkLength"] / 1609.34
    out["VHT"] = out["volume"] * out["traveltime"] / 3600.0
    cols =  [
            "linkLength",
            "linkFreeSpeed",
            "linkCapacity",
            "numberOfLanes",
            "linkModes",
            "attributeOrigId",
            "attributeOrigType",
        ]
    if index is not None:
        cols.append(index)
    return out[
        list(linkStats.columns)
        + cols
    ]
