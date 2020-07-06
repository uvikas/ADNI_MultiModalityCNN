"""
Script to get mean and standard deviation from demographic and cognitive test statistics statistics from csv files (ADNI)
"""
import numpy as np
import scipy.stats
import csv
# LMCI  SMC     CN
paths = ['/home/vikasu/Downloads/idaSearch_7_05_2020.csv', '/home/vikasu/Downloads/idaSearch_7_05_2020 (1).csv', '/home/vikasu/Downloads/idaSearch_7_05_2020 (2).csv', '/home/vikasu/Downloads/idaSearch_7_05_2020 (3).csv', '/home/vikasu/Downloads/idaSearch_7_05_2020 (4).csv']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

for path in paths:
    with open(path, 'rt') as f:
        first = True
        data = csv.reader(f)
        weight = []
        score = []
        for row in data:
            if first:
                first = False
            else:
                if row[9] != "":
                    weight.append(float(row[3]))
                    score.append(float(row[9]))
    
    print(path)
    print("Weight mean:", np.mean(weight))
    print("Weight std:", np.std(weight))

    print("Score mean:", np.mean(score))
    print("Score std:", np.std(score))
    print()