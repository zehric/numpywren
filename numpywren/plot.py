import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="plot S3 IO benchmarks")
parser.add_argument('file_name', type=str)

# DATA_SIZE = 134217728
args = parser.parse_args()
FILE_NAME = args.file_name

saved = np.load(FILE_NAME)

iops = saved['iops']
bins = saved['bins']

plt.plot(bins - min(bins), iops)
plt.ylabel("TPS")
plt.xlabel("time (s)")
plt.savefig(FILE_NAME.replace(".npz", ".png"))
