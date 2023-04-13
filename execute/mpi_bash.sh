#!/bin/bash

while getopts ":n:f:" opt; do
  case $opt in
    n)
      num_cores="$OPTARG"
      ;;
    f)
      output_file="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ -z "$num_cores" ]; then
  architecture=$(uname -m)
  if [ $architecture == "x86_64" ] || [ $architecture == "i686" ]; then
    num_cores=$(lscpu | awk '/^Core\(s\) per socket:/{print $NF}')
  elif [ $architecture == "aarch64" ]; then
    num_cores=$(cat /proc/cpuinfo | grep -E '^physical id' | uniq | wc -l)
  else
    echo "Unsupported architecture: $architecture"
    exit 1
  fi
fi

# Get the name of the executable and its arguments
shift $((OPTIND-1))
executable=$1
shift
args="$@"

echo "execute $executable $args on $num_cores cores"
# Execute the program with mpiexec and passing arguments
if [ -n "$output_file" ]; then
  mpiexec -n $num_cores $executable $args > $output_file
  echo "output written to $output_file"
else
  mpiexec -n $num_cores $executable $args
fi