#!/bin/bash

LANGUAGE=$1
GPUIDX=1
BEAMSIZE=10

# Run Training
th main.lua -dev_ref_file $CODENN_DIR/data/stackoverflow/${LANGUAGE}/dev/ref.txt -gpuidx $GPUIDX -language $1

# Run prediction
th predict.lua -encoder ${LANGUAGE}.encoder -decoder ${LANGUAGE}.decoder -beamsize $BEAMSIZE -gpuidx $GPUIDX -language ${LANGUAGE}
