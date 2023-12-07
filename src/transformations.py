import pandas as pd
import numpy as np


def getLinkStats(PTs: pd.DataFrame):
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
    linksAndTravelTimes["linkTravelTime"] = linksAndTravelTimes[
        "linkTravelTime"
    ].astype(float)
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
    return grouped


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
    PTs.drop(columns=["numPassengers", "length"], inplace=True)
    return PTs


def filterPersons(persons: pd.DataFrame):
    return persons.loc[:,
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
        ]
    ].copy()
