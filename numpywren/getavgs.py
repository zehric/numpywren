import numpy as np
import sys

# file = np.load(sys.argv[1])
# results = file.item()['results']
# print(len(results))
# avgs = []

# def getdiff(x):
#     mi = min(x, key=lambda x:x[0])
#     ma = max(x, key=lambda x:x[1])
#     return ma[1] - mi[0]

# for r in results:
#     avgs.append(getdiff(r))
    
# print(np.average(avgs))
# exit()

file = np.load(sys.argv[1])
avgs = []
results = file.item()['results']
print(len(results))
newres = []
for r in results:
    newres.append(r['time_gets'])
results = newres

def getdiff(x):
    mi = min(x, key=lambda x:x[0])
    ma = max(x, key=lambda x:x[1])
    return ma[1] - mi[0]

for r in results:
    avgs.append(getdiff(r))
    
print(np.average(avgs))
