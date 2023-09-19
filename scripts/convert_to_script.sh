#!/bin/bash

source env/bin/activate

# Convert notebook to script
jupyter nbconvert --to script DL_Beatrice_Cotti.ipynb

deactivate

git add DL_Beatrice_Cotti.py