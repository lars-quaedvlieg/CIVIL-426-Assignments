#!/bin/bash

# Loop to launch main.py with seeds from 52 to 73
for ((seed=42; seed<=73; seed++))
do
    echo "Running main.py with seed $seed"
    python main.py --seed "$seed"
done
