from typing import Optional, Dict
import os

import geopandas as gpd
import pandas as pd


class Geometry:
    def __init__(self):
        self.region = None
        self.crs = None
        self._gdf = None
        self.unit = None
        self._path = None
        self._inputcrs = None
        self._gdf = None
        self.index = None
        self._otherFiles = dict()

    @property
    def gdf(self):
        if self._gdf is None:
            self.load()
        return self._gdf

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )

    def zoneToCountyMap(self):
        return NotImplementedError("This region is not defined yet")


class SfBayGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "SFBay"
        self.crs = "epsg:26910"
        self.unit = "TAZ"
        self.index = "taz1454"
        self._path = os.path.join(os.path.dirname(__file__), '..', "geoms/sfbay-tazs-epsg-26910.shp")
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        return self._gdf.set_index(self.index)["areatype10"].to_dict()


class AustinGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Austin"
        self.crs = "epsg:26910"
        self.unit = "BG"
        self.index = "TAZ"
        self._path = "geoms/block_group_austin_26910.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Austin")


class SeattleGeometry(Geometry):
    def __init__(self, otherFiles: Optional[Dict[str, str]] = None):
        super().__init__()
        self.region = "Seattle"
        self.crs = "epsg:32048"
        self.unit = "BG"
        self.index = "OBJECTID"
        self._path = "geoms/block-groups-32048.shp"
        self._otherFiles = otherFiles

        self.load()

    def zoneToCountyMap(self):
        return self._gdf.set_index(self.index)["county"].to_dict()

    def zoneToRegionTypeMap(self):
        raise NotImplementedError("No regions defined for Seattle")

    def load(self):
        self._gdf = gpd.read_file(self._path)
        if len(self._otherFiles or []) > 0:
            for filepath, key in self._otherFiles.items():
                otherFile = pd.read_csv(filepath)
                self._gdf = pd.merge(
                    self._gdf, otherFile, left_on=self.index, right_on=key
                )
        self._gdf.rename(columns={"county_nam": "county"}, inplace=True)
