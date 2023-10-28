#!/bin/bash

for dataset in "S1" "S1-Big" "S10" "S10-Big" 
do
    eval "python3 patches_training.py ${dataset}"   
done