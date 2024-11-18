from src import input, outputDataDirectory
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.input import PilatesRunInputDirectory
from src.outputDataDirectory import PilatesOutputData, OutputDataDirectory

controlScenario = "oldBase"
template2 = "https://storage.googleapis.com/beam-core-outputs/sfbay-baseline2018-30pct-20230825"
defaultDirectory = PilatesRunInputDirectory(
                template2, [2018], 10, 1
            )
output = PilatesOutputData(
                OutputDataDirectory("output/{0}".format(controlScenario)), defaultDirectory
            )
df = output.tripModeCountPerYear.dataFrame


scenarioNames = ["base-20240711", "ESS-20240715"]
years = [2017, 2023, 2029]
asimLiteIterations = 2
beamIterations = 1
template = "/global/scratch/users/zaneedell/pilates-outputs/sfbay-atlas-{}"
folderNames = [template.format(n) for n in scenarioNames]

settings = [
    outputDataDirectory.PilatesSettings(a, b, years, asimLiteIterations, beamIterations)
    for (a, b) in zip(
        scenarioNames, folderNames
    )
]
scenario = outputDataDirectory.PilatesAnalysis(allPilatesSettings=settings)

mc = scenario.tripModeCount
mcCounty = scenario.tripModeCountByCounty

vmtByMode = scenario.vmtByMode
energyByMode = scenario.energyByMode





print('done')
# OLD STUFF
"""vmtByMode.to_csv("LKSDFJSDLFKJSDF.csv")

pops = dict()
popsByCounty = dict()
modechoices = dict()
modeChoicesByCounty = dict()
pmtByCounty = dict()
modeChoiceByPurpose = dict()
pmtByPurpose = dict()
modeVMT = dict()
for sc in pilatesScenarios:
    modeVMT[sc] = pilatesData[sc].modeVMTPerYear.dataFrame
    pops[sc] = pilatesData[sc].mandatoryLocationsByTazByYear
    modeChoiceByPurpose[sc] = (
        pilatesData[sc].asimRuns[(2010, 2)].tripModeCountByPrimaryPurpose.dataFrame
    )
    pmtByPurpose[sc] = (
        pilatesData[sc].asimRuns[(2010, 2)].tripPMTByPrimaryPurpose.dataFrame
    )
    pilatesData[sc].asimRuns[(2010, 2)].mandatoryLocationsByTaz.addMapping(
        pilatesData[sc].geometry.zoneToCountyMap(), "TAZ", "county"
    )
    popsByCounty[sc] = (
        pilatesData[sc]
        .asimRuns[(2010, 2)]
        .mandatoryLocationsByTaz.dataFrame.reset_index()
        .groupby(["county"])
        .agg({"population": sum, "jobs": sum})
        .unstack()
        .unstack(0)
    )
    modechoices[sc] = pilatesData[sc].tripModeCountPerYear.dataFrame
    pilatesData[sc].asimRuns[(2010, 2)].tripModeCountByOrigin.addMapping(
        pilatesData[sc].geometry.zoneToCountyMap(), "origin", "county"
    )
    modeChoicesByCounty[sc] = (
        pilatesData[sc]
        .asimRuns[(2010, 2)]
        .tripModeCountByOrigin.dataFrame.reset_index()
        .groupby(["county", "trip_mode"])
        .agg({"count": sum})["count"]
        .unstack()
    )
    pmtByCounty[sc] = (
        pilatesData[sc]
        .asimRuns[(2010, 2)]
        .tripPMTByOrigin.addMapping(
            pilatesData[sc].geometry.zoneToCountyMap(), "origin", "county"
        )
        .reset_index()
        .groupby(["trip_mode", "county"])
        .agg({"distanceInMiles": "sum"})["distanceInMiles"]
        .unstack(0)
    )

modeVMT["base2010"] = basePilatesData.modeVMTPerYear.dataFrame
pops["base2010"] = basePilatesData.mandatoryLocationsByTazByYear.dataFrame
modechoices["base2010"] = basePilatesData.tripModeCountPerYear.dataFrame
basePilatesData.asimRuns[(2010, 2)].tripModeCountByOrigin.addMapping(
    basePilatesData.geometry.zoneToCountyMap(), "origin", "county"
)
modeChoicesByCounty["base2010"] = (
    basePilatesData.asimRuns[(2010, 2)]
    .tripModeCountByOrigin.dataFrame.reset_index()
    .groupby(["county", "trip_mode"])
    .agg({"count": sum})["count"]
    .unstack()
)
# popsByCounty["base2010"] = (
#     basePilatesData.asimRuns[(2010, 2)]
#     .mandatoryLocationsByTaz.dataFrame.reset_index()
#     .groupby(["county"])
#     .agg({"population": sum, "jobs": sum})
#     .unstack()
#     .unstack(0)
# )

pmtByCounty["base2010"] = (
    basePilatesData.asimRuns[(2010, 2)]
    .tripPMTByOrigin.addMapping(
        basePilatesData.geometry.zoneToCountyMap(), "origin", "county"
    )
    .reset_index()
    .groupby(["trip_mode", "county"])
    .agg({"distanceInMiles": "sum"})["distanceInMiles"]
    .unstack(0)
)
modeChoiceByPurpose["base2010"] = basePilatesData.asimRuns[
    (2010, 2)
].tripModeCountByPrimaryPurpose.dataFrame
pmtByPurpose["base2010"] = basePilatesData.asimRuns[
    (2010, 2)
].tripPMTByPrimaryPurpose.dataFrame

pops = pd.concat(pops)
modechoices = pd.concat(modechoices)
modeChoicesByCounty = pd.concat(modeChoicesByCounty)
popsByCounty = pd.concat(popsByCounty)
modeChoiceByPurpose = pd.concat(modeChoiceByPurpose)
pmtByPurpose = pd.concat(pmtByPurpose)
popByScenario = pops["population"].unstack(1)[2010].unstack(0)
jobsByScenario = pops["jobs"].unstack(1)[2010].unstack(0)
print("stop")

basePop = basePilatesData.mandatoryLocationsByTazByYear.dataFrame
popChange = popByScenario.subtract(basePop.loc[2010, "population"], axis=0)

jobsChange = jobsByScenario.subtract(basePop.loc[2010, "jobs"], axis=0)


gdf = gpd.read_file("geoms/sfbay-tazs-epsg-26910.shp")
gdf = gdf.merge(
    popByScenario.add_suffix("_2020_pop"), left_on="taz1454", right_index=True
)
gdf = gdf.merge(
    basePop.loc[2010, "population"].to_frame("base_2010_pop"),
    left_on="taz1454",
    right_index=True,
)

gdf = gdf.merge(
    jobsByScenario.add_suffix("_2020_jobs"), left_on="taz1454", right_index=True
)
gdf = gdf.merge(
    basePop.loc[2010, "jobs"].to_frame("base_2010_jobs"),
    left_on="taz1454",
    right_index=True,
)

gdf["base-popchange-acre"] = gdf["base"] / gdf["gacres"]
gdf["max-telework-popchange-acre"] = gdf["max-telework"] / gdf["gacres"]
gdf["bartsv-popchange-acre"] = gdf["bartsv"] / gdf["gacres"]

popByCounty = (
    gdf[
        [
            "county",
            "base_2020_pop",
            "max-telework_2020_pop",
            "bartsv_2020_pop",
            "base_2010_pop",
        ]
    ]
    .groupby("county")
    .sum()
)
popByCounty.iloc[:, [0, 2, 1]].subtract(popByCounty["base_2020_pop"], axis=0).plot.bar()
plt.gca().get_legend().get_texts()[0].set_text("")
plt.gca().get_legend().get_texts()[1].set_text("Bart Extension")
plt.gca().get_legend().get_texts()[2].set_text("More Telework")
plt.ylabel("Difference in population from Baseline in 2020")
plt.gcf().tight_layout()

jobsByCounty = (
    gdf[
        [
            "county",
            "base_2020_jobs",
            "max-telework_2020_jobs",
            "bartsv_2020_jobs",
            "base_2010_jobs",
        ]
    ]
    .groupby("county")
    .sum()
)
jobsByCounty.iloc[:, [0, 2, 1]].subtract(
    jobsByCounty["base_2020_jobs"], axis=0
).plot.bar()
plt.gca().get_legend().get_texts()[0].set_text("")
plt.gca().get_legend().get_texts()[1].set_text("Bart Extension")
plt.gca().get_legend().get_texts()[2].set_text("More Telework")
plt.ylabel("Difference in jobs from Baseline in 2020")
plt.gcf().tight_layout()"""
