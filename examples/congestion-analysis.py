from src import input, outputDataDirectory
from src.input import SfBayGeometry
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

scenarioToLoc = {
    "newmap-jdeq-0.11": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-jdeq-0.11__2023-11-02_23-15-20_nys",
    "newmap-jdeq-0.09": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-jdeq-0.09__2023-11-02_23-11-42_puf",
    "newmap-jdeq-0.07": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-jdeq-0.07__2023-11-10_17-49-43_vgu",
    "newmap-jdeq-0.05": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-jdeq-0.07__2023-11-03_17-42-41_xlr",
    "newmap-jdeq-0.035": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-jdeq-0.035__2023-11-03_17-44-48_jed",
    "newmap-bpr-0.035": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-bpr-0.035__2023-11-02_23-12-00_wnu",
    "newmap-bpr-0.03": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-newmap-bpr-0.03__2023-11-02_23-12-04_mki",
    "oldmap-jdeq-0.09": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-oldmap-jdeq-0.09__2023-11-02_23-49-53_cwa",
    "oldmap-jdeq-0.07": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-oldmap-jdeq-0.07__2023-11-09_20-13-34_vxr",
    "oldmap-bpr-0.033": "https://storage.googleapis.com/beam-core-outputs/output/sfbay/sfbay-oldmap-bpr-0.033__2023-11-02_23-48-29_kcx",
    "newfixed-jdeq-0.07": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-newmap-jdeq-0.07__2024-01-26_00-30-11_fgz",
    # "newfixed-jdeq-0.035": "https://storage.googleapis.com/beam-core-outputs/output/testing/beamville__2023-12-01_22-37-13_wid",
}

results = dict()
moreResults = dict()

for folder, path in scenarioToLoc.items():
    beamDirectory = input.BeamRunInputDirectory(
        path, numberOfIterations=10, region="SFBay"
    )
    beamData = outputDataDirectory.BeamOutputData(
        outputDataDirectory.OutputDataDirectory("output/{0}".format(folder)),
        beamDirectory,
    )
    beamData.tazTrafficVolumes.toCsv()
    results[folder] = beamData.tazTrafficVolumes.dataFrame
    if folder.endswith("7"):
        moreResults[folder] = beamData.networkVolumesByLinkByIteration.dataFrame
    elif folder.endswith("bpr-0.035"):
        moreResults[folder] = beamData.networkVolumesByLinkByIteration.dataFrame
    elif folder.endswith("bpr-0.033"):
        moreResults[folder] = beamData.networkVolumesByLinkByIteration.dataFrame

errorIter = dict()
for path, df in moreResults.items():
    res = []
    for i in range(9):
        a = (df.iloc[:,i] - df.iloc[:,i+1]) ** 2.0
        res.append(np.sqrt(np.mean(a)))
    errorIter[tuple(path.split("-"))] = np.array(res)

byTAZ = dict()
byType = dict()
totalByType = dict()
speedByType = dict()
hoursByType = dict()
milesByType = dict()
speedTot = dict()
for name, df in results.items():
    byTAZ[tuple(name.split("-"))] = (
        df.loc[df.mph < 4.0, "VHT"]
        .groupby(["taz1454", "attributeOrigType"])
        .agg("sum")
        .unstack(fill_value=0.0)
    )
    byType[tuple(name.split("-"))] = byTAZ[tuple(name.split("-"))].sum(axis=0)
    totalByType[tuple(name.split("-"))] = (
        df.loc[:, "VHT"].groupby(["attributeOrigType"]).agg("sum")
    )
    speedByType[tuple(name.split("-"))] = (
        df.loc[:, "VMT"].groupby(["hour", "attributeOrigType"]).agg("sum")
        / df.loc[:, "VHT"].groupby(["hour", "attributeOrigType"]).agg("sum")
    ).unstack(0)
    hoursByType[tuple(name.split("-"))] = (
        df.loc[:, "VHT"].groupby(["hour", "attributeOrigType"]).agg("sum")
    ).unstack(0)
    milesByType[tuple(name.split("-"))] = (
        df.loc[:, "VMT"].groupby(["hour", "attributeOrigType"]).agg("sum")
    ).unstack(0)
    speedTot[tuple(name.split("-"))] = df.loc[:, "VMT"].groupby(["hour"]).agg(
        "sum"
    ) / df.loc[:, "VHT"].groupby(["hour"]).agg("sum")
byType = pd.DataFrame(byType).fillna(0.0).T
totalByType = pd.DataFrame(totalByType).fillna(0.0).T
speedByType = pd.concat(speedByType)
hoursByType = pd.concat(hoursByType)
milesByType = pd.concat(milesByType)
speedTot = pd.concat(speedTot)

fig, axs = plt.subplots(2, 4)
for idx, hw in enumerate(["motorway", "trunk", "primary", "secondary"]):
    speedByType.loc[pd.IndexSlice[:, :, :, hw], :].iloc[:, :30].unstack(
        [1, 2, 0]
    ).stack(0).loc[hw, ("jdeq", "0.07")].plot(ax=axs[0, idx], legend=False)
    axs[0, idx].set_ylim([10, 70])
    axs[0, idx].set_title(hw)
    (
        hoursByType.loc[pd.IndexSlice[:, :, :, hw], :]
        .iloc[:, :30]
        .unstack([1, 2, 0])
        .stack(0)
        .loc[hw, ("jdeq", "0.07")]
        / 100.0
    ).plot(ax=axs[1, idx], legend=False)
axs[0, -1].legend(["Old map", "New map"])
axs[0, 0].set_ylabel("Speed (mph)")
axs[1, 0].set_ylabel("Vehicle hours traveled (1000s)")
plt.gcf().tight_layout()


fig, axs = plt.subplots(2, 4)
for idx, hw in enumerate(["motorway", "trunk", "primary", "secondary"]):
    speedByType.loc[pd.IndexSlice[:, :, :, hw], :].iloc[:, :30].unstack(
        [1, 2, 0]
    ).stack(0).loc[hw, ("jdeq", "0.07")].plot(ax=axs[0, idx], legend=False)
    axs[0, idx].set_ylim([10, 70])
    axs[0, idx].set_title(hw)
    (
        hoursByType.loc[pd.IndexSlice[:, :, :, hw], :]
        .iloc[:, :30]
        .unstack([1, 2, 0])
        .stack(0)
        .loc[hw, ("jdeq", "0.07")]
        / 100.0
    ).plot(ax=axs[1, idx], legend=False)
axs[0, -1].legend(["New map", "Old map", "New map (minspeed)"])
axs[0, 0].set_ylabel("Speed (mph)")
axs[1, 0].set_ylabel("Vehicle hours traveled (1000s)")
plt.gcf().tight_layout()


with mpl.rc_context(
    {"axes.prop_cycle": plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 5)))}
):
    speedTot.loc[pd.IndexSlice["newmap", "jdeq", :]].unstack([0]).iloc[:30, :].plot()

with mpl.rc_context(
    {"axes.prop_cycle": plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 5)))}
):
    fig, axs = plt.subplots(1, 4)
    for idx, hw in enumerate(["motorway", "trunk", "primary", "secondary"]):
        speedByType.loc[pd.IndexSlice["newmap", "jdeq", :, hw]].T.iloc[:30, :].plot(
            ax=axs[idx], legend=False
        )
        axs[idx].set_title(hw)
        axs[idx].set_ylim([0, 70])
    axs[0].set_ylabel("Speed (mph)")
    axs[-1].legend(title="Flow Capacity")


congestedVHTbyType = pd.concat(byTAZ).fillna(0.0).groupby(level=[0, 1, 2]).sum()
portionOfTotalInJam = congestedVHTbyType.divide(totalByType.sum(axis=1), axis=0)

fig, axs = plt.subplots(1, 4)
for idx, hw in enumerate(["motorway", "trunk", "primary", "secondary"]):
    portionOfTotalInJam[hw].unstack([0, 1]).plot(
        style=["g<", "rP", "r<", "bP", "b<"], ax=axs[idx], legend=False
    )
    axs[idx].set_title(hw)
    axs[idx].set_xlabel("Flow capacity")
axs[0].set_ylabel("Portion of VHT spent at <3 mph")
axs[-1].legend(
    [
        "New map*, jdeqsim",
        "New map, bprsim",
        "New map, jdeqsim",
        "Old map, bprsim",
        "Old map, jdeqsim",
    ]
)
geometry = SfBayGeometry()

gdf = geometry._gdf.copy()
gdf["congestedOldMap"] = (
    tazInfoOldMap.loc[tazInfoOldMap.mph < 2.0]
    .value_counts("taz1454")
    .reindex_like(gdf.taz1454)
    .fillna(0.0)
)