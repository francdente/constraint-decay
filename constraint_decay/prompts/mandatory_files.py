MANDATORY_FILES_PYTHON_TEMPLATE = """## Mandatory Files

1. **`requirements.txt`** - all pip dependencies needed to run the server. Do NOT invent unexisting versions; when in doubt prefer `>=` versions.
2. **`run.sh`** - a shell script that starts the server. Must be a valid bash script.
"""

MANDATORY_FILES_NODE_TEMPLATE = """## Mandatory Files

1. **`package.json`** - all npm dependencies needed to run the server. Do NOT invent nonexistent package versions; when in doubt omit the version or use `>=`.
2. **`run.sh`** - a shell script that starts the server. Must be a valid bash script.
"""
