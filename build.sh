#!/bin/bash
# build.sh
echo "Forcing Python 3.10..."
curl -o runtime.txt https://raw.githubusercontent.com/MwailaCoding/Hackathon/main/runtime.txt?$(date +%s)
pip install -r requirements.txt