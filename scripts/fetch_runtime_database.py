"""Export the benchmark's pristine, materialized base-image database from Modal."""
from pathlib import Path
import hashlib
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'ghcr.io/snowflake-labs/data-eng-bench-base@sha256:ef92b6ef197a89ff1d8b371aaf5de19343003abc462991ddeedb1b1005e2e04b'

def main():
    output = ROOT/'data/runtime/retail.duckdb'
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    app=modal.App.lookup('superstate-graphs-environments',create_if_missing=True)
    sandbox=modal.Sandbox.create('sleep','900',app=app,image=modal.Image.from_registry(IMAGE).entrypoint([]),cpu=1,memory=1024,timeout=1200)
    partial=output.with_suffix('.partial')
    digest=hashlib.sha256()
    try:
        with sandbox.open('/app/database/retail.duckdb','rb') as remote, partial.open('wb') as local:
            while chunk:=remote.read(8*1024*1024):
                local.write(chunk)
                digest.update(chunk)
        partial.replace(output)
        receipt={'image':IMAGE,'sha256':digest.hexdigest(),'bytes':output.stat().st_size,'source':'Pristine base image after upstream fix_data.py and dbt run; before any task overlay or learner actions.'}
        output.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
    finally:
        sandbox.terminate()

if __name__=='__main__':
    main()
