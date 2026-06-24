#!/usr/bin/env sh
# Run the Lab 2 recommender from any working directory.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "${BIGDATA_LAB2_PYTHON:-}" ]; then
    PYTHON_BIN=$BIGDATA_LAB2_PYTHON
elif [ -x /root/anaconda3/envs/bigdata-lab2/bin/python ]; then
    PYTHON_BIN=/root/anaconda3/envs/bigdata-lab2/bin/python
elif command -v conda >/dev/null 2>&1; then
    exec conda run -n bigdata-lab2 python "$SCRIPT_DIR/../src/recommender.py" "$@"
else
    PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/../src/recommender.py" "$@"
