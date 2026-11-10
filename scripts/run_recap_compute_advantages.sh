#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh recap_compute_advantages --nproc 1