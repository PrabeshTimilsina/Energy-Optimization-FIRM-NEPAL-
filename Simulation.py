# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au
# Modified by: Prabesh Timilsina


import numpy as np

"""
solution — a Solution object created from a candidate capacity vector x. It carries all the decision variables (CPV, CPHP, CPHS, CInter) 
and the pre-computed arrays (MLoad, GPV, efficiencyPH, etc.). The function reads from solution and also writes results back to it at the end.

baseload — the 87,600 × 11 array of RoR hydro generation profiles in MW, pre-computed in Input.py

india_imports — a 1D array of length intervals (87,600 or a subset), giving the MW of electricity imported from India at each hour.

daily_peaking — the 87,600 × 11 array of peaking hydro profiles in MW, also pre-computed in Input.py. Non-zero only during hours 18:00–22:00.

peaking_hours — the integer 4, the width of the evening peak window

start=None and end=None — optional integer indices for slicing the simulation to a subset of hours. Normally left as None (the whole 87,600-hour period).
"""
def Reliability(solution, baseload, india_imports, daily_peaking, peaking_hours, start=None, end=None):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    ###### CALCULATE NETLOAD FOR EACH INTERVAL ######
    """
    Netload- the load that storage and imports must serve.

    solution.MLoad.sum(axis=1) — MLoad is (87,600 × 11). .sum(axis=1) sums across axis 1 (the 11 province columns) at each timestep,
    collapsing it to a 1D array of 87,600 values — the total national electricity demand in MW at each hour.

    solution.GPV.sum(axis=1) — GPV is (87,600 × 23). .sum(axis=1) sums across all 23 solar sites, 
    giving the total national solar PV generation in MW at each hour. Solar is subtracted first because it is must-take generation.

    baseload.sum(axis=1) — baseload is (87,600 × 11). .sum(axis=1) sums across all provinces giving total national RoR hydro generation in MW at each hour

    (...)[start:end] — applied after the subtraction. Slices the resulting 1D array to the simulation window. If start=None and end=None (the normal case),
    this keeps all 87,600 hours. If called from Dispatch.py with start=0, end=8760, it takes only the first year (8,760 hours).

    Netload[t] is positive when demand exceeds domestic supply
    """
    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - baseload.sum(axis=1))[start:end] \
                - india_imports # Sj-ENLoad(j, t), MW
    length = len(Netload)
    
    # needed cause Network.py needs to know the India import profile to compute how much each external node imports.
    solution.india_imports = india_imports # MW

    ###### CREATE STORAGE SYSTEM VARIABLES ######
    """
    solution.CPHP is the list of PHES power capacities per province in GW.
    Pcapacity_PH is the maximum MW rate at which PHES can charge or discharge simultaneously — the power limit of the storage system.

    solution.CPHS is the PHES energy storage capacity in GWh
    Scapacity_PH is the maximum energy the PHES reservoir can hold — the energy limit of the storage system.

    CHydro_Peaking is an array of peaking capacities per province in GW, loaded from assets_existing.csv
    Pcapacity_Peaking is the maximum MW the reservoir hydro plants can discharge simultaneously during the evening peak.

    the peaking hydro reservoir is modelled simply: it can store enough water to run at full power for exactly peaking_hours (4) hours.


    """
    Pcapacity_PH = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Pcapacity_B = sum(solution.CBP) * pow(10, 3)  # Battery power capacity (fixed exogenous constant), GW to MW
    Scapacity_PH = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    Scapacity_B = solution.CBS * pow(10, 3)  # Battery energy capacity (fixed exogenous constant), GWh to MWh
    Pcapacity_Peaking = sum(solution.CHydro_Peaking) * pow(10, 3)
    Scapacity_Peaking = peaking_hours*Pcapacity_Peaking
    efficiencyPH, efficiencyB, resolution = (solution.efficiencyPH, solution.efficiencyB, solution.resolution)

    # map(np.zeros, [length] * 5) applies np.zeros to each element of the list [length, length, length, length, length]
    # creating five separate zero arrays each of length length (87,600)
    # All seven arrays start at zero and are filled hour by hour in the loop below
    DischargePH, ChargePH, StoragePH, DischargeB, ChargeB, StorageB, DischargePeaking, StoragePeaking = map(np.zeros, [length] * 8)
    Deficit_energy, Deficit_power = map(np.zeros, [length] * 2)
    
    # daily_peaking is (87,600 × 11) — peaking generation available per province per hour. 
    # .sum(axis=1) collapses provinces to give national total peaking generation available at each hour
    # Dividing by 24 converts from a concentrated evening spike to a uniform 24-hour average rate.
    daily_peaking_divided = daily_peaking.sum(axis=1) / 24

    """
    This loop is the core of the entire model. It runs length = 87,600 times
    The chronological order is essential — each hour's storage state depends on the previous hour's state,
    oughly 7.3 billion iterations total across the full optimisation run.
    """
    for t in range(length):
        ###### INITIALISE INTERVAL ######
        Netloadt = Netload[t] # Pulls the single scalar value Netload[t]
        Storage_B_t1 = StorageB[t-1] if t>0 else 0.5 * Scapacity_B
        Storage_PH_t1 = StoragePH[t-1] if t>0 else 0.5 * Scapacity_PH # PHES storage level at the end of the previous hour — what the reservoir contains at the start of this hour, in MWh

        # Storage_Peaking_t1 is the peaking hydro reservoir level at the start of this hour, in MWh.
        # toragePeaking[t-1] + daily_peaking_divided[t] — the reservoir level from the end of the previous hour, plus the inflow during this hour. 
        # The inflow (daily_peaking_divided[t]) represents river water accumulating in the reservoir at a constant rate throughout the day.
        Storage_Peaking_t1 = StoragePeaking[t-1] + daily_peaking_divided[t] if t>0 else 0.5*Scapacity_Peaking

        # Calculate peaking discharge
        """
        This block decides how much peaking hydro to dispatch this hour.

        Case 1 — Reservoir is overfull (Scapacity_Peaking < Storage_Peaking_t1):
        The reservoir contains more than its maximum capacity
        the model forces Discharge_Peaking_t = daily_peaking_divided[t] — discharge exactly the inflow that just arrived

        Case 2 — Reservoir is within capacity:
        np.maximum(0, Netloadt) — only discharge if net load is positive (demand exceeds supply)
        np.minimum(..., Pcapacity_Peaking) — cannot discharge more than the turbine power capacity.

        Discharge_Peaking_t = np.minimum(Discharge_Peaking_t, Storage_Peaking_t1/resolution) — a third constraint: cannot discharge more energy than what is stored.

        Storage_Peaking_t = Storage_Peaking_t1 - Discharge_Peaking_t * resolution — updates the reservoir level. 
        The discharge in MW × resolution (1 hour) gives MWh consumed from the reservoir. 
        The inflow was already added at line 35, so this simple subtraction gives the end-of-hour reservoir level.
        """
        if Scapacity_Peaking < Storage_Peaking_t1:
            Discharge_Peaking_t = daily_peaking_divided[t]
        else:
            Discharge_Peaking_t = np.minimum(np.maximum(0, Netloadt), Pcapacity_Peaking)
            Discharge_Peaking_t = np.minimum(Discharge_Peaking_t, Storage_Peaking_t1/resolution)
        Storage_Peaking_t = Storage_Peaking_t1 - Discharge_Peaking_t * resolution

        # write the computed values to the output arrays
        DischargePeaking[t] = Discharge_Peaking_t
        StoragePeaking[t] = Storage_Peaking_t

        ##### UPDATE STORAGE SYSTEMS ######
        Netloadt = Netloadt - Discharge_Peaking_t # Reduce net load by peaking dispatch, peaking hydro always goes first, and PHES operates on whatever residual remains.

        # Battery is capped by the largest contingency and acts before PHES.
        Discharge_B_t = min(max(0, Netloadt), Pcapacity_B, Storage_B_t1 / resolution)
        Charge_B_t = min(-1 * min(0, Netloadt), Pcapacity_B, (Scapacity_B - Storage_B_t1) / efficiencyB / resolution)
        Storage_B_t = Storage_B_t1 - Discharge_B_t * resolution + Charge_B_t * resolution * efficiencyB

        DischargeB[t] = Discharge_B_t
        ChargeB[t] = Charge_B_t
        StorageB[t] = Storage_B_t

        diff1 = Netloadt - Discharge_B_t + Charge_B_t

        Discharge_PH_t = min(max(0, diff1), Pcapacity_PH, Storage_PH_t1 / resolution)

        # PHES then handles the remaining energy balancing.
        # -1 * min(0, diff1) — only charge if diff1 < 0
        # Pcapacity_PH — cannot charge faster than the pump power capacity
        # (Scapacity_PH - Storage_PH_t1) / efficiencyPH / resolution — cannot charge more than the available empty space in the reservoir. 
        # (Scapacity_PH - Storage_PH_t1) is the unfilled space in MWh. Dividing by efficiencyPH (0.8) accounts for losses
        Charge_PH_t = min(-1 * min(0, diff1), Pcapacity_PH, (Scapacity_PH - Storage_PH_t1) / efficiencyPH / resolution)
        
        """
        Storage_PH_t1 — the starting level
        - Discharge_PH_t * resolution — subtract the energy discharged. 
        + Charge_PH_t * resolution * efficiencyPH — add the energy stored. Electricity consumed for pumping (Charge_PH_t MW × 1 hour = MWh) is multiplied by efficiencyPH (0.8) to get the MWh.
        """
        Storage_PH_t = Storage_PH_t1 - Discharge_PH_t * resolution + Charge_PH_t * resolution * efficiencyPH

        # Write the three computed scalars into their respective output arrays at position
        DischargePH[t] = Discharge_PH_t
        ChargePH[t] = Charge_PH_t
        StoragePH[t] = Storage_PH_t

        diff2 = diff1 - Discharge_PH_t + Charge_PH_t
        
        ###### DETERMINE DEFICITS ######
        """
        diff1 <= 0 (balanced or surplus):
        Both deficit types are zero. The system met all demand with peaking hydro + PHES + solar + RoR + imports. No action needed.

        Discharge_PH_t == Pcapacity_PH (power-limited deficit):
        PHES discharged at its absolute maximum power — the turbines ran flat out. 
        Yet demand was still not met. The storage reservoir probably had energy left in it (Storage_PH_t1 > 0), 
        but the turbines could not release it fast enough. This is a power deficit

        Discharge_PH_t == Storage_PH_t1 / resolution (energy-limited deficit):
        PHES tried to discharge as much as needed, was within its power cap, but ran out of stored energy
        """
        if diff2 <= 0:
            Deficit_energy[t] = 0
            Deficit_power[t] = 0
        elif ((Discharge_PH_t == Pcapacity_PH) and (Discharge_B_t == Pcapacity_B)):
            Deficit_energy[t] = 0
            Deficit_power[t] = diff2
        elif ((Discharge_PH_t == Storage_PH_t1 / resolution) and (Discharge_B_t == Storage_B_t1 / resolution)):
            Deficit_energy[t] = diff2
            Deficit_power[t] = 0
        elif ((Discharge_PH_t == Pcapacity_PH) and (Discharge_B_t == Storage_B_t1 / resolution)):
            Deficit_energy[t] = diff2
            Deficit_power[t] = diff1 - diff2
        elif ((Discharge_PH_t == Storage_PH_t1 / resolution) and (Discharge_B_t == Pcapacity_B)):
            Deficit_energy[t] = diff1 - diff2
            Deficit_power[t] = diff2

    """
    After the loop has processed all 87,600 hours, this line adds the two deficit type arrays element-wise. 
    Deficit[t] is the total MW of unmet demand at hour t, regardless of whether it was energy-limited or power-limited. 
    This is the array that the penalty function PenDeficit in Optimisation.py uses
    """
    Deficit = Deficit_energy + Deficit_power
    Netload = Netload - DischargePeaking #  subtracting the peaking hydro dispatch at each hour. This post-loop update produces what the net load looks like after peaking hydro but before PHES
    Spillage = -1 * np.minimum(Netload + ChargePH + ChargeB - DischargePH - DischargeB, 0) # Spillage is the surplus renewable generation that could not be stored or used — it was simply wasted.
    Spillage = np.maximum(0, Spillage)  # Clip tiny floating-point negatives to zero (physically meaningless)

    ###### ERROR CHECKING ######
    assert 0 <= int(np.amax(StoragePH)) <= Scapacity_PH, 'Storage below zero or exceeds max storage capacity'
    assert 0 <= int(np.amax(StorageB)) <= Scapacity_B, 'StorageB below zero or exceeds max storage capacity'
    assert np.amin(Deficit) > -0.1, 'DeficitD below zero'
    assert np.amin(Spillage) >= 0, 'Spillage below zero'

    ###### UPDATE SOLUTION OBJECT ######
    solution.DischargePH, solution.ChargePH, solution.StoragePH, solution.DischargePeaking, solution.StoragePeaking = (DischargePH, ChargePH, StoragePH, DischargePeaking, StoragePeaking)
    solution.DischargeB, solution.ChargeB, solution.StorageB = (DischargeB, ChargeB, StorageB)
    solution.Deficit_energy, solution.Deficit_power, solution.Deficit, solution.Spillage = (Deficit_energy, Deficit_power, Deficit, Spillage)

    # Returns six arrays to the caller (Optimisation.py
    return Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage

if __name__ == '__main__':
    from Input import *
    from Network import Transmission 

    suffix = "_Super_existing_2_True_True.csv"
    Optimisation_x = np.genfromtxt('Results/Optimisation_resultx{}'.format(suffix), delimiter=',') #  loads the 39-element optimal solution vector from the CSV file saved by Optimisation.py
    
    # Initialise the optimisation, wraps it into a Solution object
    S = Solution(Optimisation_x)

    CIndia = np.nan_to_num(np.array(S.CInter)) # converts the import capacities to a safe numeric array. This is exactly the same setup as the start of F(x) in Optimisation.py.

    # Simulation with only baseload
    """
    These lines replicate the entire 3-simulation logic from F(x) in Optimisation.py, using the loaded solution. 
    The logic is identical — Simulation 1 with zero imports, check penalties, clip to get realistic imports, 
    Simulation 3 with realistic imports. The purpose is to reproduce the exact system state that the optimiser used, 
    enabling detailed analysis that was not saved during the fast optimisation loop.

    One difference: the standalone block calls Reliability(S, ..., daily_peaking=daily_peaking) without the peaking_hours keyword argument, 
    while Optimisation.py passes it explicitly. This is because the standalone block is slightly out of sync with the function signature — 
    the code still works because daily_peaking and peaking_hours are available as module-level variables from from Input import *.
    """
    Deficit_energy1, Deficit_power1, Deficit1, DischargePH1, DischargePeaking1, Spillage1 = Reliability(S, baseload=baseload, india_imports=np.zeros(intervals), daily_peaking=daily_peaking, peaking_hours=peaking_hours)
    Max_deficit1 = np.reshape(Deficit1, (-1, 8760)).sum(axis=-1) # MWh per year
    PIndia = Deficit1.max() * pow(10, -3) # GW

    GIndia = resolution * (Max_deficit1).max()

    # Penalise only under-sized interconnectors
    PenPower = max(0, PIndia - CIndia.sum()) * pow(10,3)
    PenEnergy = 0
    
    # Simulation with baseload, all existing capacity
    Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=np.ones(intervals) * CIndia.sum() * pow(10,3), daily_peaking=daily_peaking, peaking_hours=peaking_hours)

    # Deficit penalty function
    PenDeficit = max(0, Deficit.sum() * resolution - S.allowance)

    # India import profile
    # Determine clipping bounds based on Input.py flags
    low = -CIndia.sum() * pow(10,3) if allow_exports else 0
    high = CIndia.sum() * pow(10,3) if allow_imports else 0
    # Allow negative values for export when permitted; otherwise constrain to allowed directions
    india_imports = np.clip(Deficit1 - Spillage1, low, high) # MW

    # Hard cap: limit gross imports to gross_import_cap_fraction of annual demand, and gross exports to gross_export_cap_fraction
    from Optimisation import gross_import_cap_fraction, apply_gross_import_cap, gross_export_cap_fraction, apply_gross_export_cap
    annual_demand_mwh = MLoad.sum() * resolution / years  # MWh per year
    india_imports = apply_gross_import_cap(india_imports, gross_import_cap_fraction, annual_demand_mwh, years)
    india_imports = apply_gross_export_cap(india_imports, gross_export_cap_fraction, annual_demand_mwh, years)

    # Simulation using the existing capacity generation profiles - required for storage average annual discharge
    Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=india_imports, daily_peaking=daily_peaking, peaking_hours=peaking_hours)

    # Discharged energy from storage systems
    GPHES = DischargePH.sum() * resolution / years * pow(10,-6) # TWh per year
    # Battery sizing (fixed exogenous constant, 30-minute hold for frequency regulation)
    GBattery = S.DischargeB.sum() * resolution / years * pow(10,-6)  # Average annual discharge, TWh

    # Transmission capacity calculations
    """
    Replicates the network and LCOE calculations from Optimisation.py for the loaded solution. 
    Note the cost array here is slightly different from Optimisation.py's cost array — 
    it has 4 fewer elements (no wind placeholder, different ordering of hydro/zero terms)
    """
    TDC = Transmission(S, output=True) if 'Super' in node else np.zeros((intervals, len(TLoss))) # TAC: TAC(t, k), MW
    CAC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CAC(k), MW to GW

    # Transmission penalty functionS
    PenDC = 0

    # Average annual electricity generated by existing capacity
    GHydro = resolution * (baseload.sum() + DischargePeaking.sum()) / efficiencyPH / years
    
    # Split into gross imports and gross exports for separate cost factors
    GIndia_imp = resolution * india_imports[india_imports > 0].sum() / years  # gross import MWh/year
    GIndia_exp = resolution * (-india_imports[india_imports < 0].sum()) / years  # gross export MWh/year (positive value)
    GIndia = resolution * india_imports.sum() / years  # net exchange MWh/year (kept for legacy use)

    # Levelised cost of electricity calculation
    factor_dict = dict(np.genfromtxt('Data/factor_hvac.csv', dtype=None, delimiter=',', encoding=None))
    transmission_cost = np.array([factor_dict['SPKP'], factor_dict['KPLP'], factor_dict['LPGP'], factor_dict['GPBP'], factor_dict['BPMP'], factor_dict['EPMP'], factor_dict['TISP'], factor_dict['GILP'], factor_dict['MIMP'], factor_dict['KIEP']])
    cost = (
        factor_dict['PV'] * sum(S.CPV)
        + factor_dict['India-imp'] * GIndia_imp * pow(10, -6)  # cost of importing from India
        - factor_dict['India-exp'] * GIndia_exp * pow(10, -6)  # revenue from exporting to India (reduces cost)
        + factor_dict['PHP'] * sum(S.CPHP)
        + factor_dict['PHS'] * S.CPHS
        + factor_dict['BP'] * sum(S.CBP)
        + factor_dict['BS'] * S.CBS
        + factor_dict['PHES-VOM'] * GPHES
        + factor_dict['B-VOM'] * GBattery
        + (transmission_cost * CAC).sum()
        + factor_dict['ACPV'] * sum(S.CPV)
        + factor_dict['Hydro'] * GHydro * pow(10, -6)
    ) # $b p.a.
    loss = np.sum(abs(TDC), axis=0) * TLoss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss) 

    print(LCOE, energy, loss, GIndia_imp * pow(10,-6) * factor_dict['India-imp'] / (GIndia_imp*pow(10,-9)) if GIndia_imp != 0 else 0, (GHydro) * pow(10, -6) * factor_dict['Hydro'] / ((GHydro)*pow(10,-9)) if GHydro != 0 else 0, len(CAC))
