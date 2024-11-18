from typing import Optional, List, Dict, Set, Union
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Constraint:
    name: str
    definition: str

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, self.definition))


class Table:
    def __init__(self, df: pd.DataFrame, name: str, index_col: Optional[List[str]] = None):
        self.index_col = index_col or []
        self.name = name
        self.__rawSchema = self.__defineBasicSchema(df.head(0), name, index_col)
        self.__fieldToIdx: Dict[str, int] = {}
        self.__fieldToColumn: Dict[str, 'Column'] = {}
        self.__constraints: Set[Constraint] = set()
        self.__parseRawSchema()

        # Add primary key constraint if index columns were specified
        if self.index_col:
            pk_columns = ', '.join(f'"{col}"' for col in self.index_col)
            self.addConstraint(Constraint(
                name=f"pk_{self.name}",
                definition=f"PRIMARY KEY ({pk_columns})"
            ))

    @staticmethod
    def __defineBasicSchema(df: pd.DataFrame, table_name: str, indices: Optional[List[str]]) -> List[str]:
        return pd.io.sql.get_schema(df.reset_index(drop=True), table_name, keys=indices).split('\n')

    def __parseRawSchema(self):
        for idx, full_string in enumerate(self.__rawSchema):
            if (idx > 0) and not full_string.lstrip().startswith("CONSTRAINT") and not full_string.startswith(")"):
                # Remove leading whitespace and trailing comma
                cleaned_string = full_string.strip().rstrip(',')
                if '"' in cleaned_string:
                    name = cleaned_string.split('"')[1]
                    dtype = cleaned_string.split('"')[2].lstrip()
                    self.__fieldToIdx[name] = idx
                    self.__fieldToColumn[name] = Column(
                        self,
                        name,
                        dtype,
                        primaryKey=name in self.index_col
                    )

    def addConstraint(self, constraint: Constraint):
        """Add a new constraint to the table"""
        self.__constraints.add(constraint)

    def getColumn(self, name: str) -> 'Column':
        """Get a column by name"""
        return self.__fieldToColumn.get(name)

    def getColumns(self, names: List[str]) -> List['Column']:
        """Get multiple columns by name"""
        return [self.getColumn(name) for name in names]

    def listColumns(self) -> List[str]:
        """List all column names"""
        return list(self.__fieldToColumn.keys())

    def updateColumnType(self, column_name: str, new_type: str, using: Optional[str] = None) -> 'Column':
        """
        Update the data type of a specific column.

        Args:
            column_name: Name of the column to modify
            new_type: New SQL data type for the column
            using: Optional USING clause for type conversion

        Returns:
            Updated Column object

        Raises:
            ValueError: If column doesn't exist
        """
        if column_name not in self.__fieldToColumn:
            raise ValueError(f"Column '{column_name}' does not exist in table '{self.name}'")

        column = self.__fieldToColumn[column_name]
        column.columnType = new_type
        return column

    def toSqlCommand(self) -> str:
        """Generate the CREATE TABLE command with all constraints in DuckDB style"""
        # Start with table name
        sql_parts = [f'CREATE TABLE "{self.name}" (']

        # Add columns
        column_definitions = []
        for name, column in self.__fieldToColumn.items():
            column_def = column.toSqlDefinition()
            column_definitions.append(f"    {column_def}")

        # Add constraints
        constraint_definitions = []
        for constraint in self.__constraints:
            constraint_definitions.append(
                f"    {constraint.definition}"
            )

        # Combine all parts
        all_definitions = column_definitions + constraint_definitions
        sql_parts.extend([',\n'.join(all_definitions)])
        sql_parts.append(');')

        return '\n'.join(sql_parts)


class Column:
    def __init__(
            self,
            parentTable: Table,
            columnName: str,
            columnType: str,
            primaryKey: bool = False,
            nullable: bool = True,
            unique: bool = False
    ):
        self.parentTable = parentTable
        self.columnName = columnName
        self.columnType = columnType.strip()
        self.isPrimaryKey = primaryKey
        self.isNullable = nullable and not primaryKey  # Primary keys can't be nullable
        self.isUnique = unique or primaryKey  # Primary keys are always unique

    def toSqlDefinition(self) -> str:
        """Generate the SQL column definition in DuckDB style"""
        parts = [f'"{self.columnName}" {self.columnType}']

        if not self.isNullable:
            parts.append("NOT NULL")
        if self.isUnique and not self.isPrimaryKey:  # Add UNIQUE constraint if needed
            parts.append("UNIQUE")

        return " ".join(parts)


class ForeignKeyConstraint:
    def __init__(
            self,
            fromColumns: Union[Column, List[Column]],
            toColumns: Union[Column, List[Column]],
    ):
        # Convert single columns to lists for uniform handling
        self.fromColumns = [fromColumns] if isinstance(fromColumns, Column) else fromColumns
        self.toColumns = [toColumns] if isinstance(toColumns, Column) else toColumns

        # Validate column counts match
        if len(self.fromColumns) != len(self.toColumns):
            raise ValueError("Number of source and target columns must match")

        # Generate constraint name using all column names
        from_cols = '_'.join(col.columnName for col in self.fromColumns)
        to_cols = '_'.join(col.columnName for col in self.toColumns)
        self.name = f"fk_{self.fromColumns[0].parentTable.name}_{from_cols}_" \
                    f"{self.toColumns[0].parentTable.name}_{to_cols}"

        # Add the constraint to the parent table
        self.updateSchema()

    def updateSchema(self):
        """Add the foreign key constraint to the parent table"""
        from_cols = ', '.join(f'"{col.columnName}"' for col in self.fromColumns)
        to_cols = ', '.join(f'"{col.columnName}"' for col in self.toColumns)

        constraint_def = (
            f"FOREIGN KEY ({from_cols}) "
            f"REFERENCES \"{self.toColumns[0].parentTable.name}\" ({to_cols}) "
        )
        self.fromColumns[0].parentTable.addConstraint(Constraint(self.name, constraint_def))


class Schema:
    def __init__(self):
        self.__tableNameToTable: Dict[str, Table] = {}
        self.__foreignKeys: List[ForeignKeyConstraint] = []

    def __setitem__(self, key: str, value: Table):
        """Add a table to the schema"""
        if not isinstance(value, Table):
            raise ValueError("Value must be a Table instance")
        self.__tableNameToTable[key] = value

    def __getitem__(self, key: str) -> Table:
        """Get a table by name"""
        return self.__tableNameToTable[key]

    def addForeignKey(
            self,
            fromTable: str,
            fromColumns: Union[str, List[str]],
            toTable: str,
            toColumns: Union[str, List[str]]
    ) -> ForeignKeyConstraint:
        """
        Add a foreign key relationship between tables.

        Args:
            fromTable: Name of the referencing table
            fromColumns: Column(s) in the referencing table (single string or list)
            toTable: Name of the referenced table
            toColumns: Column(s) in the referenced table (single string or list)
            onDelete: ON DELETE behavior
            onUpdate: ON UPDATE behavior
        """
        if fromTable not in self.__tableNameToTable or toTable not in self.__tableNameToTable:
            raise ValueError("Tables must exist in schema")

        # Convert single column names to lists
        from_cols = [fromColumns] if isinstance(fromColumns, str) else fromColumns
        to_cols = [toColumns] if isinstance(toColumns, str) else toColumns

        # Get Column objects
        from_columns = self.__tableNameToTable[fromTable].getColumns(from_cols)
        to_columns = self.__tableNameToTable[toTable].getColumns(to_cols)

        # Check if all columns exist
        if not all(from_columns) or not all(to_columns):
            raise ValueError("All columns must exist in tables")

        fk = ForeignKeyConstraint(from_columns, to_columns)
        self.__foreignKeys.append(fk)
        return fk

    def foreignKeys(self) -> List[ForeignKeyConstraint]:
        """Get all foreign key constraints in the schema"""
        return self.__foreignKeys.copy()

    def toSql(self) -> str:
        """Generate complete SQL for creating all tables in the schema"""
        return "\n\n".join(
            table.toSqlCommand()
            for table in self.__tableNameToTable.values()
        )

if __name__ == "__main__":
    # Create sample DataFrames
    orders_df = pd.DataFrame({
        'order_year': [2023, 2023, 2024],
        'order_number': [1, 2, 1],
        'customer_id': [101, 102, 103]
    })

    order_items_df = pd.DataFrame({
        'item_year': [2023, 2023, 2023],
        'item_order': [1, 1, 2],
        'product_id': [1, 2, 1]
    })

    # Create schema
    schema = Schema()

    # Add tables with composite primary keys
    schema['orders'] = Table(orders_df, 'orders', index_col=['order_year', 'order_number'])
    schema['order_items'] = Table(order_items_df, 'order_items', index_col=['item_year', 'item_order'])

    # Add composite foreign key
    schema.addForeignKey(
        fromTable='order_items',
        fromColumns=['item_year', 'item_order'],
        toTable='orders',
        toColumns=['order_year', 'order_number']
    )

    # Generate SQL
    print(schema.toSql())