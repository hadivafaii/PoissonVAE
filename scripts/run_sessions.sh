#!/bin/bash


num_fits=${1}

# Check if num_gpus is provided
if [ -z "${2}" ]; then
  # Count the number of available GPUs
  num_gpus=$(nvidia-smi --list-gpus | wc -l)
else
  num_gpus=${2}
fi

# Find max index of fits for a specific GPU
function find_max_fit {
  local gpu=${1}
  local max_fit=-1
  local current_fit
  for file in "$(uname -n)"-cuda"${gpu}"-fit*.txt; do
    if [[ $file =~ "$(uname -n)"-cuda${gpu}-fit([0-9]+)\.txt ]]; then
      current_fit=${BASH_REMATCH[1]}
      (( current_fit > max_fit )) && max_fit=$current_fit
    fi
  done
  echo $(( max_fit + 1 ))  # fit indices start at 0
}

# Main loop
for ((gpu=0; gpu<num_gpus; gpu++)); do
  if [ -z "${num_fits}" ]; then
    num_fits=$(find_max_fit "${gpu}")
  fi
  for ((fit=0; fit<num_fits; fit++)); do
    name="cuda${gpu}-fit${fit}"
    name="$(uname -n)-${name}"
    if ! screen -list | grep -q "${name}"; then
      screen -dmS "${name}"
      echo "${name}: session created."
    else
      echo "Screen ${name} already exists —> executing."
    fi
    screen -S "${name}" -p 0 -X stuff "bash ${name}.txt\n"
  done
done
