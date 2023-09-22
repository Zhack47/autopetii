#!/usr/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
docker build -t autopet_baseline "$SCRIPTPATH"
