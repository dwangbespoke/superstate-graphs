#!/bin/bash
set -euo pipefail
cp -a /solution/oracle/. /app/sg_project/
cd /app/sg_project
dbt run --project-dir /app/sg_project --profiles-dir /app/sg_project --no-use-colors
