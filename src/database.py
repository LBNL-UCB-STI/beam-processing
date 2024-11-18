from typing import Optional, List, Dict

import pandas as pd





class Table:
    def __init__(self, df: pd.DataFrame, name: str, index_col: Optional[List[str]] = None):
        self.index_col = index_col or []
        self.name = name
        self.__rawSchema = self.__defineBasicSchema(df.head(0), name, index_col)
        self.__fieldToIdx = {}
        self.__fieldToColumn = {}
        self.__parseRawSchema()

    @staticmethod
    def __defineBasicSchema(df: pd.DataFrame, table_name, indices) -> List[str]:
        return pd.io.sql.get_schema(df.reset_index(), table_name, keys=indices).split('\n')

    def __parseRawSchema(self):
        for idx, fullString in enumerate(self.__rawSchema):
            if (idx > 0) and not fullString.lstrip.startswith("CONSTRAINT") and not fullString.startswith(")"):
                name = fullString.split('"')[1]
                dtype = fullString.split('"')[2].lstrip()[:-1]
                self.__fieldToIdx[name] = idx
                self.__fieldToColumn = Column(self, name, dtype, primaryKey=name in self.index_col)

    def toSqlCommand(self) -> str:
        # Update raw schema here. This is a placeholder
        return '\n'.join(self.__rawSchema)


class Column:
    def __init__(self, parentTable: Table, columnName: str, columnType: str, primaryKey=False):
        self.parentTable = parentTable
        self.columnName = columnName
        self.columnType = columnType
        self.isPrimaryKey = primaryKey


class ForeignKeyConstraint:
    def __init__(self, firstColumn: Column, secondColumn: Column):
        self.__firstColumn = firstColumn
        self.__secondColumn = secondColumn
        self.updateSchemas()

    def updateSchemas(self):
        # placeholder to update the raw schema in the parent tables for each column

class Schema:
    def __init__(self):
        self.__tableNameToTable = {}

    def __setitem__(self, key: str, value: Table):
        self.__tableNameToTable[key] = value

    def foreignKeys(self) -> List[ForeignKeyConstraint]:
        # Placeholder to loop over ForeignKeyConstraints in the tables in __tableNameToTable
        return []