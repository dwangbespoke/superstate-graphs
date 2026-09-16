#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
if python3 /tests/check_task.py; then echo 1 > /logs/verifier/reward.txt; else exit 1; fi
