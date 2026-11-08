# Step-by-step analysis to decide the dispatch of flexible energy resources
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au
# Modified by: Prabesh Timilsina

from Input import *
from Simulation import Reliability

import datetime as dt
from multiprocessing import Pool, cpu_count

def Flexible(instance):
    """Energy source of high flexibility"""

    year, x = instance

    S = Solution(x)

    #startidx = int((24 / resolution) * (dt.datetime(year, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    #endidx = int((24 / resolution) * (dt.datetime(year+1, 1, 1) - dt.datetime(firstyear, 1, 1)).days)
    startidx = int((24 / resolution) * (year-firstyear) * 365)
    endidx = int((24 / resolution) * (year+1-firstyear) * 365)

    Fcapacity = np.nan_to_num(np.array(S.CInter)).sum() * pow(10, 3) # GW to MW
    flexible = Fcapacity * np.ones(endidx - startidx)

    for i in range(0, endidx - startidx):
        flexible[i] = 0
        Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=flexible, daily_peaking=daily_peaking, peaking_hours=peaking_hours, start=startidx, end=endidx) # Sj-EDE(t, j), MW
        #print(year, i, Deficit.sum(), DischargePond.sum(), baseload.sum())
        if Deficit.sum() * resolution > 0.1:
            flexible[i] = Fcapacity - DischargePeaking[i] - DischargePH[i] - S.DischargeB[i]

    Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=flexible, daily_peaking=daily_peaking, peaking_hours=peaking_hours, start=startidx, end=endidx) # Required after updating final interval of flexible

    flexible = np.clip(flexible - S.Spillage, 0, None)

    print('Dispatch works on', year)

    return flexible

def Analysis(x,suffix):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Dispatch starts at', starttime)

    # Multiprocessing
    pool = Pool(processes=min(cpu_count(), finalyear - firstyear + 1))
    #print(cpu_count(), finalyear, firstyear)
    instances = map(lambda y: [y] + [x], range(firstyear, finalyear + 1))
    Dispresult = pool.map(Flexible, instances)
    pool.terminate()

    Flex = np.concatenate(Dispresult)
    np.savetxt('Results/Dispatch_Flexible' + suffix, Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('Dispatch took', endtime - starttime)

    from Statistics import Information
    Information(x, Flex)

    return True

if __name__ == '__main__':
    suffix="_Super_existing_20_True_True.csv"
    capacities = np.genfromtxt('Results/Optimisation_resultx'+suffix, delimiter=',')
    Analysis(capacities,suffix)
