import pandas as pd


def filterPersons(persons: pd.DataFrame):
    cols_to_keep = [
        "earning",
        "is_worker",
        "is_student",
        "household_id",
        "school_zone_id",
        "home_zone_id",
        "age",
        "work_zone_id",
        "workplace_zone_id",
        "distance_to_school",
        "distance_to_work",
        "TAZ",
        "home_x",
        "home_y",
        "sex",
        "pemploy",
    ]
    return persons.loc[
        :,
        [c for c in persons.columns if c in cols_to_keep],
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
    cols_to_keep = [
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
    ]
    return trips.loc[
        :,
        [c for c in trips.columns if c in cols_to_keep],
    ].copy()


def filterTours(tours: pd.DataFrame):
    if tours is not None:
        return tours[
            [
                "person_id",
                "tour_type",
                "tour_category",
                "number_of_participants",
                "destination",
                "origin",
                "household_id",
                "start",
                "end",
                "duration",
                "composition",
                "destination_logsum",
                "tour_mode",
                "mode_choice_logsum",
                "atwork_subtour_frequency",
                "parent_tour_id",
                "stop_frequency",
                "primary_purpose",
            ]
        ].copy()
    else:
        return pd.DataFrame()
