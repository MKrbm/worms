#!/bin/bash

for seed in {25..120}
do
    old_path="/Users/keisuke/Documents/projects/todo/worms/job/FF1D_sps_3/FF1D/seed_${seed}_uni.log.bak"
    new_path="/Users/keisuke/Documents/projects/todo/worms/job/FF1D_sps_3/FF1D/seed_${seed}_uni.log"
    
    if [ -f "$old_path" ]; then
        mv "$old_path" "$new_path"
        echo "Renamed: $old_path to $new_path"
    else
        echo "File not found: $old_path"
    fi
done