#!/bin/bash

runs=10000
program="../build/benchmarks/fmm-test"
seed="1"
output_file_fmm="bench_fmm"
output_file_openblas="bench_openblas"
params="32 32 32"

# Clear the output files before starting the loops
> "$output_file_openblas"
> "$output_file_fmm"

echo "Running openblas interations"
for ((i=1; i<=$runs; i++)); do
    # echo "Running openblas iteration $i/$runs"
    $program "openblas" $seed $params >> $output_file_openblas
done

echo "Running fmm interations"
for ((i=1; i<=$runs; i++)); do
    # echo "Running fmm iteration $i/$runs"
    $program "fmm" $seed $params >> $output_file_fmm
done