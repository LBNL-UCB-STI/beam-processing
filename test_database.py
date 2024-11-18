import duckdb
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, MetaData, Integer, String, Column
import src.database_v2 as db
# con = duckdb.connect("file.db")
ds = pd.HDFStore("/Users/zaneedell/git/PILATES/pilates/urbansim/data/urbansim-inputs-custom_mpo_06197001_model_data.h5","r")

hh = ds['/households'].reset_index()

hh['year'] = 2010
hh['scenario'] = "baseline"
hh['runId'] = 0
hh['recent_mover'] = hh['recent_mover'] == "1"
hh['hispanic_head'] = hh['hispanic_head'] == "yes"
hh['hh_children'] = hh['hh_children'] == "yes"
hh['sf_detached'] = hh['sf_detached'] == "yes"
hh['hh_seniors'] = hh['hh_seniors'] == "yes"


# hh['block_id'] = hh['block_id']

# engine = create_engine("postgresql+psycopg://postgres:beamcore@localhost:5432/firsttestserver")
# with engine.connect() as connection:
#     # result = connection.execute("SET role = replicate;")
#     hh.head(0).to_sql("households", con=connection, if_exists="replace", method='multi')

per = ds['/persons'].reset_index()
per['work_zone_id'] = per['work_zone_id'].astype(np.int64)
per['school_block_id'] = per['school_block_id'].astype(np.int64)
per['school_zone_id'] = per['school_zone_id'].astype(np.int64)
per['work_block_id'] = per['work_block_id'].astype(np.int64)
per['school_taz'] = per['school_taz'].astype(np.int64)
per['workplace_taz'] = per['workplace_taz'].astype(np.int64)

per['year'] = 2010
per['scenario'] = "baseline"
per['runId'] = 0

with duckdb.connect("file.db") as con:
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    schema = db.Schema()
    schema["households"] = db.Table(hh, "households", index_col=["household_id", "year", "scenario", "runId"])
    schema["persons"] = db.Table(per, "persons", index_col=["person_id", "year", "scenario", "runId"])
    schema["persons"].updateColumnType("person_id", "LONG")
    schema["persons"].updateColumnType("household_id", "LONG")
    schema["persons"].updateColumnType("work_block_id", "LONG")
    schema["persons"].updateColumnType("school_block_id", "LONG")
    schema["households"].updateColumnType("household_id","LONG")
    schema.addForeignKey(
        fromTable='persons',
        fromColumns=["household_id", "year", "scenario", "runId"],
        toTable='households',
        toColumns=["household_id", "year", "scenario", "runId"]
    )
    con.execute(schema.toSql())
    con.execute("INSERT INTO households SELECT * FROM hh")
    con.execute("INSERT INTO persons SELECT * FROM per")
    print("PAUSE")

# per.head(0).to_sql("persons", con=engine, if_exists="replace")

from sqlalchemy.dialects import postgresql

def get_sqlalchemy_type(pandas_type):
    strType = str(pandas_type)
    if strType == 'int64':
        return Integer
    elif strType == 'object':
        return String
    else:
        return pandas_type
    # Add more mappings as needed

for col, dtype in hh.dtypes.items():
    table.append_column(Column(col, get_sqlalchemy_type(dtype)))


print('stop')
a = MetaData()
a.reflect(bind=engine)

inspector = inspect(engine)
schemas = inspector.get_schema_names()

for schema in schemas:
    print("schema: %s" % schema)
    for table_name in inspector.get_table_names(schema=schema):
        for column in inspector.get_columns(table_name, schema=schema):
            print("Column: %s" % column)