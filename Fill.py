# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:17:37 2020
@author: cheng + tim
"""
# Modified by: Prabesh Timilsina


from Input import *
from Simulation import Reliability
from Optimisation import gross_import_cap_fraction, apply_gross_import_cap, gross_export_cap_fraction, apply_gross_export_cap
import numpy as np
import datetime as dt

def fill_deficit(deficit,india_imports,india_limit,india_annual,impflag,eff,step):
    idx = np.where(deficit > 0)[0]
    for idd, i in np.ndenumerate(idx):
        d = deficit[i]
        t = i
        count = 0
        #print("--------------------")
        #print(idd, " of ", len(idx))
        try:
            while d > 0 and t >= 0 and count < step:
                #print("t = ",t)
                year = t // 8760
                start = year * 8760
                end = (year+1) * 8760
                if t == i - 1:
                    d = d / eff
                if impflag:
                    remaining = india_annual - sum(india_imports[start:end])
                    assert remaining >= 0
                    hydro_c = min(india_imports[t] + d, india_limit, india_imports[t] + remaining)
                    d = d - (hydro_c - india_imports[t])
                    india_imports[t] = hydro_c
                    if remaining == 0:
                        
                        t = start - 1
                    else:
                        idxx = np.where(india_imports < india_limit)[0]
                        t = sorted([i for i in idxx if i < t])[-1]
                count += 1
        except:
            continue
    return india_imports

def save(imp,suffix):
    np.savetxt('Results/Dispatch_IndiaImports' + suffix, imp, fmt='%f', delimiter=',', newline='\n', header='India Imports')
    
def maxx(x):
    return np.reshape(x, (-1, 8760)).sum(axis=-1).max()/1e6

def mean(x):
    return x.sum()/years/1e6

def Analysis(optimisation_x,suffix):
    starttime = dt.datetime.now()
    print('Deficit fill starts at', starttime)

    S = Solution(optimisation_x)
    
    Deficit_energy1, Deficit_power1, Deficit1, DischargePH1, DischargePeaking1, Spillage1 = Reliability(S, baseload=baseload, india_imports=np.zeros(intervals), daily_peaking=daily_peaking, peaking_hours=peaking_hours)
    Max_deficit1 = np.reshape(Deficit1, (-1, 8760)).sum(axis=-1) # MWh per year
    PIndia = Deficit1.max() * pow(10, -3) # GW

    GIndia = resolution * (Max_deficit1).max() / efficiencyPH    

    print("GIndia_max:", GIndia/1e6)
    
    GIndia2 = Deficit1.sum() / years / 0.8
    
    print("GIndia_mean:", GIndia2/1e6)

    Indiamax = energy*pow(10,9)
    annual_demand_mwh = MLoad.sum() * resolution / years  # MWh per year
    Indiamax = gross_import_cap_fraction * annual_demand_mwh  # Hard cap: 20% of annual demand in MWh
    print("IMPORTS")
    print("------------------------------")
    # Determine clipping bounds based on scenario flags in Input.py
    low = -sum(S.CInter) * 1e3 if allow_exports else 0
    high = sum(S.CInter) * 1e3 if allow_imports else 0
    # Initialize imports allowing export where there is spillage (surplus => negative export)
    india_imports = np.clip(Deficit1 - Spillage1, low, high)
    # Hard cap: limit gross imports to gross_import_cap_fraction of annual demand, and gross exports to gross_export_cap_fraction
    india_imports = apply_gross_import_cap(india_imports, gross_import_cap_fraction, annual_demand_mwh, years)
    india_imports = apply_gross_export_cap(india_imports, gross_export_cap_fraction, annual_demand_mwh, years)
    # Re-run reliability to get updated Deficit/Spillage given the initial export/import allocation
    Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=india_imports, daily_peaking=daily_peaking, peaking_hours=peaking_hours)
    # Fill any remaining positive deficit using imports only if imports are allowed
    if allow_imports:
        imp = fill_deficit(Deficit, india_imports, sum(S.CInter) * 1e3, Indiamax, True, 0.8, 168)
        imp = apply_gross_import_cap(imp, gross_import_cap_fraction, annual_demand_mwh, years)
        imp = apply_gross_export_cap(imp, gross_export_cap_fraction, annual_demand_mwh, years)
    else:
        imp = india_imports
    Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=imp, daily_peaking=daily_peaking, peaking_hours=peaking_hours)
    print("India generation:", maxx(imp))
    print("Remaining deficit:", Deficit.sum()/1e6)
    if allow_imports:
        step = 1
        while Deficit.sum() > allowance*years and step < 50:
            imp = fill_deficit(Deficit,imp,sum(S.CInter)*1e3,Indiamax,True,0.8,168)
            imp = apply_gross_import_cap(imp, gross_import_cap_fraction, annual_demand_mwh, years)
            imp = apply_gross_export_cap(imp, gross_export_cap_fraction, annual_demand_mwh, years)
            Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=imp, daily_peaking=daily_peaking, peaking_hours=peaking_hours)
            step += 1
    print("India generation max:", maxx(imp))
    print("India generation mean:", mean(imp))
    print("Remaining deficit final:", Deficit.sum()/1e6)

    save(imp,suffix)

    endtime = dt.datetime.now()
    print('Deficit fill took', endtime - starttime)

    from Statistics import Information
    Information(optimisation_x,imp)

    return True

if __name__=='__main__':
    suffix = "_Super_existing_2_True_True.csv"
    optimisation_x = np.genfromtxt('Results/Optimisation_resultx{}'.format(suffix), delimiter=',')
    Analysis(optimisation_x,suffix)
