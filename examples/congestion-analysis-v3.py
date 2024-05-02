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
    # "simp": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07__2024-02-14_16-07-00_lsg",
    # "simp2": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07__2024-02-20_21-00-10_xba",
    "simpmulti7": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07-storage-5__2024-03-27_09-49-22_nca",
    "simpwarmest7": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07-storage-5__2024-03-18_23-27-24_irs",
    "simpwarmestest7": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07-storage-5__2024-03-21_09-18-13_lrl",
    "simpwarm7": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07-storage-5__2024-03-13_21-11-13_hvh",
    "simpwarmer7": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.07-storage-5__2024-03-15_23-08-52_bzo",
    "simpwarm6": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.06-storage-5__2024-03-12_04-52-34_svh",
    "simpwarmer6": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.06-storage-5__2024-03-18_22-56-31_xwy",
    # "simpwarmer6": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.06-storage-5__2024-03-07_15-21-13_voe",
    # "simpwarm55": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.055-storage-5__2024-03-07_15-12-01_sbn",
    # "simpwarm5": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-simp-jdeq-0.05-storage-5-multijdeqsim__2024-03-08_15-23-58_vvg",
    # "psimp": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-psimp-jdeq-0.07-0.5__2024-02-15_19-59-31_qbf",
    # "res": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-res-jdeq-0.07__2024-02-14_17-26-14_gcj",
    # "psimpnores": "https://storage.googleapis.com/beam-core-outputs/output/testing/sfbay-psimpnores-jdeq-0.07-0.5__2024-02-15_23-13-23_kja",
}

results = dict()
moreResults = dict()
linkData = dict()

for folder, path in scenarioToLoc.items():
    if folder == "simpwarm55":
        n = 8
    elif folder == "simpwarmer6":
        n = 7
    elif folder == "simpwarmestest7":
        n = 9
    elif folder == "simpmulti7":
        n = 2
    else:
        n = 10
    beamDirectory = input.BeamRunInputDirectory(
        path, numberOfIterations=n, region="SFBay"
    )
    beamData = outputDataDirectory.BeamOutputData(
        outputDataDirectory.OutputDataDirectory("output/{0}".format(folder)),
        beamDirectory,
    )
    results[folder] = beamData.tazTrafficVolumes.dataFrame
    linkData[folder] = beamData.labeledLinkStatsFile.dataFrame
    moreResults[folder] = beamData.networkVolumesByLinkByIteration.dataFrame


# for path, df in linkData.items():
#     df["VHTperMile"] = df["VHT"] / df["length"] * 1609.34
#     df = df.loc[df["VHT"] > 100.0, :]
#     df["mph"] = df["VMT"] / df["VHT"]
#     srtd = df.sort_values("VHT", ascending=False)
#     smaller = srtd.loc[~srtd.reset_index()["link"].duplicated().values, :]
#     net = beamData.labeledNetwork.dataFrame
#     net = net.merge(smaller, on="link")
#     gdf = gpd.GeoDataFrame(net, geometry=getGeometry(net))

"""
res = results['simpwarm55']
look = res.groupby(['hour','attributeOrigType']).apply(lambda x: x['VMT'].sum() / x['VHT'].sum()).unstack()
look.iloc[:25,:].plot()

ld = linkData["simpwarm55"]
ld['delay'] = ld['VHT'] / (ld['volume'] * ld['length'] / ld['freespeed'] /3600.0)
byLink = ld.groupby('link').agg({'delay':'sum', "attributeOrigType":'first',"attributeOrigId":'first','linkCapacity':'first','numberOfLanes':'first','volume':'sum'}).sort_values('delay', ascending=False)
"""


def getGeometry(df):
    geometry = df.apply(
        lambda x: LineString(
            [
                Point(x.fromLocationX, x.fromLocationY),
                Point(x.toLocationX, x.toLocationY),
            ]
        ),
        axis=1,
    )
    return geometry


def getPoint(df, x, y):
    geometry = df.apply(lambda row: Point(row[x], row[y]), axis=1)
    return geometry


errorIter = dict()
totTT = dict()
for path, df in moreResults.items():
    if (path == "simpwarm55") | (path == "simpwarmer6"):
        n = 7
    elif path == "simpmulti7":
        n = 2
    else:
        n = 10
    res = []
    for i in range(n - 1):
        a = (df.iloc[:, i] - df.iloc[:, i + 1]) ** 2.0
        res.append(np.sqrt(np.mean(a)))
    errorIter[tuple(path.split("-"))] = np.array(res)
    totTT[tuple(path.split("-"))] = df.sum(axis=0)

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

df = linkData["simpwarm"]
df["delay"] = df["VHT"] - (df["volume"] * df["length"] / df["freespeed"] / 3600.0)
linkTots = df.groupby("link").agg(
    {
        "VHT": sum,
        "VMT": sum,
        "delay": sum,
        "linkFreeSpeed": "first",
        "attributeOrigId": "first",
        "attributeOrigType": "first",
        "linkCapacity": "first",
    }
)
look = linkTots.sort_values("delay", ascending=False).head(200)

fig, axs = plt.subplots(2, 4)
for idx, hw in enumerate(["motorway", "trunk", "primary", "secondary"]):
    speedByType.loc[pd.IndexSlice[:, :, :, :, hw], :].iloc[:, :30].unstack(
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
