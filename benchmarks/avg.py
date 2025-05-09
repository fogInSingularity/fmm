import numpy as np

filename_openblas = "bench_openblas"
filename_fmm = "bench_fmm"

with open(filename_openblas, 'r') as f:
    lines = f.readlines()

openblas_data = []
for line in lines:
    openblas_data.append(int(line))

with open(filename_fmm, 'r') as f:
    lines = f.readlines()

fmm_data = []
for line in lines:
    fmm_data.append(int(line))

print("openblas:", np.mean(openblas_data), "+-", np.std(openblas_data))
print("fmm:", np.mean(fmm_data), "+-", np.std(fmm_data))