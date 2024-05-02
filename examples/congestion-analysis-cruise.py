import urllib

from src import input, outputDataDirectory
from src.input import SfBayGeometry
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from shapely.geometry import Point, LineString
import os
import matplotlib

matplotlib.use("TkAgg")
os.chdir("../")
scenarioToLoc = {
    "Phase 1 BAU": "https://storage.googleapis.com/beam-core-outputs/sfbay-cruise-BAU_UpdatedRHSkims_dynamicFleets2/beam/year-2020-iteration-2",
    "Phase 1 Baseline": "https://storage.googleapis.com/beam-core-outputs/sfbay-cruise-SAVBaseline_1_FINAL/beam/year-2020-iteration-2",
    "Phase 2 BAU": "gs://cruise-outputs/sfbay_cruise_SAVBaseline_1_phase2_66/beam/year-2020-iteration-2",
}

byTazHourType = dict()
byTazHourClass = dict()
byLinkHour = dict()
byLink = dict()

for folder, path in scenarioToLoc.items():
    beamDirectory = input.BeamRunInputDirectory(
        path, numberOfIterations=0, region="SFBay"
    )
    beamData = outputDataDirectory.BeamOutputData(
        outputDataDirectory.OutputDataDirectory("output/{0}".format(folder)),
        beamDirectory,
    )
    byTazHourType[folder] = beamData.tazTrafficVolumes.dataFrame
    relabeled = beamData.tazTrafficVolumes.dataFrame.reset_index()
    relabeled["npmrds_class"] = relabeled["attributeOrigType"].map(
        {
            "secondary": "minor arterial",
            "tertiary": "major collector",
            "residential": "local",
            "tertiary_link": "major collector",
            "unclassified": "minor collector",
            "secondary_link": "minor arterial",
            "trunk": "other fwy or expwy",
            "primary": "other principal arterial",
            "motorway_link": "interstate",
            "motorway": "interstate",
            "primary_link": "other principal arterial",
            "trunk_link": "other fwy or expwy",
        }
    )
    relabeled = relabeled.groupby(["taz1454", "hour", "npmrds_class"]).agg(
        {"VMT": "sum", "VHT": "sum"}
    )
    relabeled["mph"] = relabeled["VMT"] / relabeled["VHT"]
    byTazHourClass[folder] = relabeled
    byLink[folder] = beamData.labeledLinkStatsFile.dataFrame
    byLinkHour[folder] = beamData.networkVolumesByLink.dataFrame


totalByType = dict()
speedByType = dict()
hoursByType = dict()
milesByType = dict()
speedTot = dict()
for name, df in byTazHourClass.items():
    totalByType[name] = (
        df.loc[:, "VHT"]
        .groupby(["npmrds_class"])
        .agg("sum")
        .loc[
            [
                "interstate",
                "other fwy or expwy",
                "other principal arterial",
                "minor arterial",
                "major collector",
                "minor collector",
            ]
        ]
    )
    speedByType[name] = (
        (
            df.loc[:, "VMT"].groupby(["hour", "npmrds_class"]).agg("sum")
            / df.loc[:, "VHT"].groupby(["hour", "npmrds_class"]).agg("sum")
        )
        .unstack(0)
        .loc[
            [
                "interstate",
                "other fwy or expwy",
                "other principal arterial",
                "minor arterial",
                "major collector",
                "minor collector",
            ]
        ]
    )
    hoursByType[name] = (
        (df.loc[:, "VHT"].groupby(["hour", "npmrds_class"]).agg("sum"))
        .unstack(0)
        .loc[
            [
                "interstate",
                "other fwy or expwy",
                "other principal arterial",
                "minor arterial",
                "major collector",
                "minor collector",
            ]
        ]
    )
    milesByType[name] = (
        (df.loc[:, "VMT"].groupby(["hour", "npmrds_class"]).agg("sum"))
        .unstack(0)
        .loc[
            [
                "interstate",
                "other fwy or expwy",
                "other principal arterial",
                "minor arterial",
                "major collector",
                "minor collector",
            ]
        ]
    )
    speedTot[name] = df.loc[:, "VMT"].groupby(["hour"]).agg("sum") / df.loc[
        :, "VHT"
    ].groupby(["hour"]).agg("sum")
totalByType = pd.DataFrame(totalByType).fillna(0.0).T
speedByType = pd.concat(speedByType)
hoursByType = pd.concat(hoursByType)
milesByType = pd.concat(milesByType)
speedTot = pd.concat(speedTot)

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
speedByType.loc["Phase 1 BAU"].T.iloc[4:25, :].plot(ax=axs[0])
speedByType.loc["Phase 1 Baseline"].T.iloc[4:25, :].plot(ax=axs[1])
speedByType.loc["Phase 2 BAU'"].T.iloc[4:25, :].plot(ax=axs[2])
axs[0].set_title("Phase 1 BAU")
axs[1].set_title("Phase 1 Baseline")
axs[2].set_title("Phase 2 BAU")
axs[0].legend().set_title("Link type")
axs[1].legend().set_visible(False)
axs[2].legend().set_visible(False)
axs[0].set_ylabel("Speed (mph)")
plt.gcf().tight_layout()
