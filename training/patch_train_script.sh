#!/bin/bash

for model in "base" "big"
do
    for dropout in "dropout" "no_dropout"
    do 
        for dataset in "S1" "S1-Big" "S10" "S10-Big" 
        do
            eval "python3 patches_training.py ${dataset} ${model} ${dropout}"   
        done
    done
done