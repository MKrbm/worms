#!/bin/bash

# Test function to echo the generated values
test_echo() {
  local Jx=$1
  local Jy=$2
  local H=$3
  echo "Generated Values: Jx=$Jx, Jy=$Jy, H=$H"
}

# Generate the list of Jx, Jy, and H values
generate_values() {
  for i in $(seq -20 4 20); do
    for j in $(seq -20 4 20); do
      for k in 0 5 10; do
        Jx=$(awk -v val=$i 'BEGIN{ printf "%.2f\n", val/10 }')
        Jy=$(awk -v val=$j 'BEGIN{ printf "%.2f\n", val/10 }')
        H=$(awk -v val=$k 'BEGIN{ printf "%.2f\n", val/10 }')
        test_echo $Jx $Jy $H
      done
    done
  done
}

# Call the generate_values function
generate_values
