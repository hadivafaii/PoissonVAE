#!/bin/bash

device=${1}
dataset=${2}
model=${3}
archi=${4}

if [ -z "${device}" ]; then
  read -rp "enter device idx: " device
fi
if [ -z "${dataset}" ]; then
  read -rp "enter dataset: " dataset
fi
if [ -z "${model}" ]; then
  read -rp "enter model type: " model
fi
if [ -z "${archi}" ]; then
  read -rp "enter architecture type: " archi
fi

# Shift to remove the first two positional arguments
# then combine the remaining arguments into one
shift 4
args="${*}"

root="Dropbox/git/_PoissonVAE"
root="${HOME}/${root}"
cd "${root}" || exit

fit="python3 -m vae.train_vae \
  '${device}' \
  '${dataset}' \
  '${model}' \
  '${archi}' \
   ${args}"
eval "${fit}"

printf '**************************************************************************\n'
printf "Done! —————— device = 'cuda:${device}' —————— (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'