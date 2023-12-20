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
    linksAndTravelTimes["links"] = linksAndTravelTimes[
        "links"
    ].astype(pd.Int64Dtype())
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


def labelNetworkWithTaz(network: pd.DataFrame, TAZ: gpd.GeoDataFrame):
    gdf = gpd.GeoDataFrame(
        network,
        geometry=gpd.points_from_xy(
            network["toLocationX"], network["toLocationY"], crs="epsg:26910"
        ),
    ).sjoin(TAZ.loc[:, ["taz1454", "geometry"]], how="left")
    return pd.DataFrame(gdf.drop(columns=["geometry", "index_right"]))


def mergeLinkstatsWithNetwork(linkStats: pd.DataFrame, network: pd.DataFrame):
    linkStats['VMT'] = linkStats['volume'] * linkStats['length'] / 1609.34
    linkStats['VHT'] = linkStats['volume'] * linkStats['traveltime'] / 3600.
    out = linkStats.merge(network, left_on='link', right_index=True)
    return out[list(linkStats.columns) + ['linkLength', 'linkFreeSpeed', 'linkCapacity', 'numberOfLanes',
       'linkModes', 'attributeOrigId', 'attributeOrigType', 'taz1454']]
