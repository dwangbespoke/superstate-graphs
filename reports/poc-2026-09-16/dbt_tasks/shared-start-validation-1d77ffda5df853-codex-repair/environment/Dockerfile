FROM ghcr.io/snowflake-labs/data-eng-bench-base@sha256:ef92b6ef197a89ff1d8b371aaf5de19343003abc462991ddeedb1b1005e2e04b
WORKDIR /app
COPY project/ /app/sg_project/
ENV DB_TYPE=duckdb DUCKDB_PATH=/app/database/retail.duckdb
