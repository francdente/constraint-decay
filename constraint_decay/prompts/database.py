DATABASE_SQLITE_TEMPLATE = """## Database

Use **SQLite** as the database.
The database schema MUST be created automatically on server startup.
"""

DATABASE_POSTGRES_TEMPLATE = """## Database

Use **PostgreSQL** as the database.
The database schema MUST be created automatically on server startup.

An instance of Postgres 16 is already running and listening on port 5432.
The database configuration you MUST use for the connection is:
- username=user
- password=password
- db=conduit
- host=postgres
- port=5432
"""