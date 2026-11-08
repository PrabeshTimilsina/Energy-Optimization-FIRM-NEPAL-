# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au
# Modified by: Prabesh Timilsina


import numpy as np

"""
solution — the Solution object carrying all simulation results from Simulation.py, and all input data
output=False — controls whether the disaggregated provincial arrays are written back onto solution
When True (called from Statistics.py for the final run), all provincial arrays are stored so Statistics.py can write them to CSV files.
"""
def Transmission(solution, domestic_only=False, export_only=False, output=False):
    """TDC = Network.Transmission(S)"""

    """
    Nodel — array of 11 node name strings: ['SP', 'KP', 'LP', 'GP', 'BP', 'MP', 'EP', 'TI', 'GI', 'MI', 'KI']
    PVl — array of 23 strings mapping each solar site to its province: ['SP','SP','SP','KP','KP',...]
    Interl — array of 4 strings for external import nodes: ['TI','GI','MI','KI']
    Hydrol — array of 7 strings mapping each hydro record to its province: ['SP','KP','LP','GP','BP','MP','EP']
    intervals = 87600 — total number of hourly timesteps
    nodes = 11 — total number of nodes including the 4 external India nodes. 
    inters = 4 — number of external interconnections
    """
    Nodel, PVl, Interl, Hydrol = (solution.Nodel, solution.PVl, solution.Interl, solution.Hydrol)
    intervals, nodes, inters = (solution.intervals, solution.nodes, len(Interl))
    
    CHydro_Peaking = solution.CHydro_Peaking # extracts the array of peaking hydro power capacities per province in GW
    """
    np.tile(CHydro_Peaking, (intervals, 1)) — takes the 7-element CHydro_Peaking array and stacks it intervals times vertically, 
    creating a (87600 × 7) matrix where every row is the same capacity array. Dividing by sum(CHydro_Peaking) converts each row to fractions. 
    The result is peakingfactor, shape (87600 × 7).  
    if Karnali has 2.0 GW out of a total 5.0 GW, its fraction is 0.40.

    MPeaking_long — the provincial peaking hydro generation matrix
    solution.DischargePeaking is a 1D array of 87,600 values (national total peaking discharge in MW at each hour, from Simulation.py).
    np.tile(solution.DischargePeaking, (len(CHydro_Peaking), 1)) stacks it 7 times horizontally, creating a (7 × intervals) matrix 
    where every row is the national DischargePeaking. .transpose() flips it to (intervals × 7).
    Multiplying element-wise by peakingfactor allocates each hour's national discharge to provinces proportionally.
    """
    peakingfactor = np.tile(CHydro_Peaking, (intervals, 1)) / sum(CHydro_Peaking) if sum(CHydro_Peaking) != 0 else 0 
    MPeaking_long = np.tile(solution.DischargePeaking, (len(CHydro_Peaking), 1)).transpose() * peakingfactor 

    MPV, MBaseload, MPeaking = map(np.zeros, [(nodes, intervals)] * 3) # creates three zero matrices each of shape (nodes × intervals) = (11 × 87600)
    for i, j in enumerate(Nodel): # iterates over all 11 nodes. i is the row index (0 through 10), j is the node name string (e.g., 'SP', 'KP', ..., 'KI')
        MPV[i, :] = solution.GPV[:, np.where(PVl==j)[0]].sum(axis=1) # fills row i with the total solar PV generation from all sites belonging to province j.
        # MWind[i, :] = solution.GWind[:, np.where(Windl==j)[0]].sum(axis=1)
        MBaseload[i, :] = solution.baseload[:, np.where(Hydrol==j)[0]].sum(axis=1) # same logic but for RoR hydro
        MPeaking[i, :] = MPeaking_long[:, np.where(Hydrol==j)[0]].sum(axis=1)
    MPV, MBaseload, MPeaking = (MPV.transpose(), MBaseload.transpose(), MPeaking.transpose()) # Sij-GPV(t, i), Sij-GWind(t, i), MW
    
    MLoad = solution.MLoad # EOLoad(t, j), MW, Retrieves the (87600 × 11) demand matrix from the solution object

    """
    defactor — a (87600 × 11) matrix of load fractions. Each element is the fraction of national demand that province j represents at hour t
    MLoad.sum(axis=1) — sums across the 11 province columns at each timestep, giving a 87,600-element array of total national demand per hour.
    [:, None] — adds a new axis to convert the 1D array (87600,) into a 2D column vector (87600 × 1). 
    This enables NumPy broadcasting: dividing (87600 × 11) by (87600 × 1) divides each row of MLoad by that row's sum, which is the same operation on every column.

    MDeficit — a (87600 × 11) matrix. MDeficit[t,j] is the MW of unmet demand attributed to province j at hour t
    np.tile(solution.Deficit, (nodes, 1)) — takes the 87,600-element national deficit array from Simulation.py and stacks it nodes (11) times to create a (11 × 87600) matrix where every row is the same national deficit profile
    """
    defactor = MLoad / MLoad.sum(axis=1)[:, None]
    MDeficit = np.tile(solution.Deficit, (nodes, 1)).transpose() * defactor # MDeficit: EDE(j, t)

    """
    M_minFactors = np.full((intervals, nodes), pow(10,-9)) — creates a (87600 × 11) matrix filled entirely with 1×10⁻⁹ (one billionth). 
    This tiny floor value is added to solar generation before computing spillage fractions. The reason: at night, MPV is zero everywhere, 
    making the spillage fraction calculation 0/0 (dividing zero solar by zero total solar). This would produce NaN values. 
    Adding a tiny positive floor to every element means the ratio is always defined — each province gets an infinitesimally equal share of any nighttime spillage.
    """
    M_minFactors = np.full((intervals, nodes), pow(10,-9)) # Matrix of 10^(-9) required to distribute spillage between nodes when no solar generation
    MPW = MPV + M_minFactors # adds the floor to the solar generation matrix.
    spfactor = np.divide(MPW, MPW.sum(axis=1)[:, None], where=MPW.sum(axis=1)[:, None]!=0) # (87600 × 11) matrix of fractions that sum to 1.0 across provinces at each hour.
    MSpillage = np.tile(solution.Spillage, (nodes, 1)).transpose() * spfactor # MSpillage: ESP(j, t), exactly the same construction as MDeficit but using spfactor (solar-proportional) instead of defactor

    """
    CIndia — builds an 11-element array of India connection capacities in GW
    nodes - len(solution.CInter) = 11 - 4 = 7, so np.array([0]*7) creates 7 zeros for the internal provincial nodes
    np.nan_to_num(np.array(solution.CInter)) converts the 4 India link capacities to a safe numeric array.
    np.append(...) concatenates them: [0,0,0,0,0,0,0, CInter_TI, CInter_GI, CInter_MI, CInter_KI]
    """
    CIndia = np.append(np.array([0]*(nodes - len(solution.CInter))), np.nan_to_num(np.array(solution.CInter))) # GW
    CPHP = solution.CPHP # list of 11 PHES power capacities per node in GW, External India nodes have zero PHES capacity.
    pcfactor = np.tile(CPHP, (intervals, 1)) / sum(CPHP) if sum(CPHP) != 0 else 0 # a (87600 × 11) proportional allocation matrix for PHES, analogous to peakingfactor
    MDischargePH = np.tile(solution.DischargePH, (nodes, 1)).transpose() * pcfactor # MDischarge: DPH(j, t), (87600 × 11) matrices allocating national PHES discharge to proveinces
    MChargePH = np.tile(solution.ChargePH, (nodes, 1)).transpose() * pcfactor # MCharge: CHPH(j, t), (87600 × 11) matrices allocating national PHES charge to provinces

    CBP = solution.CBP
    bfactor = np.tile(CBP, (intervals, 1)) / sum(CBP) if sum(CBP) != 0 else 0
    MDischargeB = np.tile(solution.DischargeB, (nodes, 1)).transpose() * bfactor
    MChargeB = np.tile(solution.ChargeB, (nodes, 1)).transpose() * bfactor

    india_imports = solution.india_imports # MW,  retrieves the 87,600-element import time series written by Simulation.py
    if CIndia.sum() == 0: # No import
        ifactor = np.tile(CIndia, (intervals, 1)) # creates an (87600 × 11) matrix of all zeros — all provinces get zero imports.
    else: 
        ifactor = np.tile(CIndia, (intervals, 1)) / CIndia.sum() # tiles the 11-element CIndia array intervals times to make (87600 × 11), then divides each row by the total import capacity. The result is an (87600 × 11) fraction matrix where only the four India-connection columns (TI, GI, MI, KI) are non-zero.
    MIndia = np.tile(india_imports, (nodes, 1)).transpose() * ifactor # (87600 × 11) matrix allocating national imports to nodes. Most columns (SP through EP) are zero — internal provinces do not directly receive India imports.

   # efactor = np.array([0,0,0,0,0,0,0,1,0,0,0])
    #ch2factor = np.array([0,1,0,0,0,0,0,0,0,0,0])
   # MExport = np.tile(solution.indiaExportProfiles, (nodes, 1)).transpose() * efactor
   # MHydro_CH2 = np.tile(solution.indiaExportProfiles, (nodes, 1)).transpose() * ch2factor

    """  CHydro_nodes = np.zeros(nodes)
    for j in range(0,len(Nodel)):
        CHydro_nodes[j] = solution.CHydro_max[expl==Nodel[j]].sum()
    expfactor = np.tile(CHydro_nodes, (intervals, 1)) / sum(CHydro_nodes) if sum(CHydro_nodes) != 0 else 0
    MSpillage_exp = np.tile(solution.Spillage, (nodes, 1)).transpose() * expfactor # MSpillage: ESP(j, t)
    MSpillage = np.zeros((nodes, intervals)).transpose() """

    """ BaseloadDomestic = MChargePH.sum(axis=1) + MLoad.sum(axis=1) - MIndia.sum(axis=1) - MDischargePH.sum(axis=1) - MDeficit.sum(axis=1)
    BaseloadDomestic[BaseloadDomestic < 0] = 0
    BaseloadExports = MBaseload.sum(axis=1) - BaseloadDomestic
    BaseloadExports[BaseloadExports < 0] = 0
    b1factor = np.divide(MBaseload, MBaseload.sum(axis=1)[:, None], where=MBaseload.sum(axis=1)[:, None]!=0)
    MBaseload_exp  = np.tile(BaseloadExports, (nodes, 1)).transpose() * b1factor """

    """ SolarDomestic = MChargePH.sum(axis=1) + MLoad.sum(axis=1) - MIndia.sum(axis=1) - MBaseload.sum(axis=1) - MDischargePH.sum(axis=1) - MDeficit.sum(axis=1)
    SolarDomestic[SolarDomestic < 0] = 0
    SolarExports = MPV.sum(axis=1) - SolarDomestic        
    SolarExports[SolarExports < 0] = 0
    s1factor = np.divide(MPV, MPV.sum(axis=1)[:, None], where=MPV.sum(axis=1)[:, None]!=0)
    MPV_exp  = np.tile(SolarExports, (nodes, 1)).transpose() * s1factor """
        
   # WindDomestic = MChargePH.sum(axis=1) + MLoad.sum(axis=1) - MIndia.sum(axis=1) - MBaseload.sum(axis=1) - MPV.sum(axis=1) - MDischargePH.sum(axis=1) - MDeficit.sum(axis=1)
   # WindDomestic[WindDomestic < 0] = 0
   # WindExports = MWind.sum(axis=1) - WindDomestic      
    #WindExports[WindExports < 0] = 0  
    #w1factor = np.divide(MWind, MWind.sum(axis=1)[:, None], where=MWind.sum(axis=1)[:, None]!=0)
    #MWind_exp  = np.tile(WindExports, (nodes, 1)).transpose() * w1factor

    """ PeakingDomestic = MChargePH.sum(axis=1) + MLoad.sum(axis=1) - MPV.sum(axis=1) - MIndia.sum(axis=1) - MBaseload.sum(axis=1) - MDischargePH.sum(axis=1) - MDeficit.sum(axis=1)
    PeakingDomestic[PeakingDomestic < 0] = 0
    PeakingExports = MPeaking.sum(axis=1) - PeakingDomestic
    PeakingExports[PeakingExports < 0]
    p1factor = np.divide(MPeaking, MPeaking.sum(axis=1)[:, None], where=MPeaking.sum(axis=1)[:, None]!=0)
    MPeaking_exp  = np.tile(PeakingExports, (nodes, 1)).transpose() * p1factor """

    if domestic_only:
        MImport = MLoad + MChargePH + MSpillage \
                - MPV - MIndia - MBaseload - MPeaking - MDischargePH - MDeficit # EIM(t, j), MW,  It is a (87600 × 11) matrix where each element MImport[t, j] is the net MW that province j must import from the rest of the grid at hour t.
        MImport = MLoad + MChargePH + MChargeB + MSpillage \
                - MPV - MIndia - MBaseload - MPeaking - MDischargePH - MDischargeB - MDeficit
    # disabled export model
    """ else: 
        MImport = MLoad + MChargePH + MSpillage_exp + MSpillage \
              - MPV - MIndia - MBaseload - MPeaking - MDischargePH - MDeficit  # EIM(t, j), MW """
    
    coverage = solution.coverage # coverage is the list of active nodes for this run. For node='Super', it is all 11 nodes. For a single-province run, it would be just 1 node.
    if len(coverage) > 1: #  only compute multi-node transmission flows if there are at least 2 nodes.

       # ['SPKP', 'KPLP', 'LPGP', 'GPBP', 'BPMP', 'EPMP', 'TISP', 'GILP', 'MIMP', 'KIEP']

        """
        Why compute external nodes first? Because the external India connection nodes (TI, GI, MI, KI) are boundary conditions for the internal chain. 
        Their flows do not depend on any internal corridor calculation — they depend only on their own MImport. 
        All internal corridor flows are computed as cumulative sums starting from these boundaries.
        """
        #these are the external node so calculation woulde be −1 × 𝑅𝑒𝑞𝑢𝑖𝑟𝑒𝑑 𝐼𝑚𝑝𝑜𝑟𝑡 (𝑡, GI) 
        TISP =  1 * MImport[:, np.where(Nodel=='TI')[0][0]] if 'TI' in coverage else np.zeros(intervals)
        GILP =  1 * MImport[:, np.where(Nodel=='GI')[0][0]] if 'GI' in coverage else np.zeros(intervals)
        KIEP =  1 * MImport[:, np.where(Nodel=='KI')[0][0]] if 'KI' in coverage else np.zeros(intervals)
        MIMP =  -1 * MImport[:, np.where(Nodel=='MI')[0][0]] if 'MI' in coverage else np.zeros(intervals)
       
    
        """
        SPKP is the power flow on the corridor between Sudurpaschim (SP) and Karnali (KP) at each hour. 
        The sign convention: positive SPKP = power flowing from SP toward KP (westward generation surplus). 
        Negative SPKP = power flowing from KP toward SP
        """
        #Imports into inner internal nodes 
        SPKP = -1 * MImport[:, np.where(Nodel=='SP')[0][0]] - TISP if 'SP' in coverage else np.zeros(intervals)

        #we need to calculate node to node connection so we need to substract in between node.
        """
        MImport[:, KP_index] — KP's net import requirement. Note: this is +MImport_KP (positive), unlike SP which used -1 × MImport_SP. 
        This sign difference is intentional and encodes the direction convention
        """
        KPLP = MImport[:, np.where(Nodel=='KP')[0][0]] - SPKP  if 'KP' in coverage else np.zeros(intervals) # EPMP is the flow on the Koshi/Eastern (EP) to Madhesh (MP) corridor
        EPMP = -1 * MImport[:, np.where(Nodel=='EP')[0][0]] - KIEP if 'EP' in coverage else np.zeros(intervals)
        LPGP = -1 * MImport[:, np.where(Nodel=='LP')[0][0]] - GILP - KPLP if 'LP' in coverage else np.zeros(intervals)
        GPBP = 1 * MImport[:, np.where(Nodel=='GP')[0][0]] - LPGP if 'GP' in coverage else np.zeros(intervals) # GPBP is the flow on the Gandaki (GP) to Bagmati (BP) corridor.
        BPMP = -1 * MImport[:, np.where(Nodel=='BP')[0][0]] - GPBP  if 'BP' in coverage else np.zeros(intervals)
        
        #Check the final node
        # BPMP1 — an independent calculation of the BP→MP corridor flow, starting from the eastern end of the network (MP node) rather than the western end.
        # BPMP1 = MImport_MP - MIMP - EPMP.
        BPMP1 = 1 *  MImport[:, np.where(Nodel=='MP')[0][0]] - MIMP - EPMP if 'MP' in coverage else np.zeros(intervals)
        """
        This check is the Kirchhoff's current law verification — it confirms that power is conserved at every node at every hour. 
        If this check fails, the candidate solution is invalid and should be skipped rather than crashing the optimizer.
        """
        bpmp_error = abs(BPMP - BPMP1).max()
        if bpmp_error > 0.5:
            if output:
                print('BPMP Error', bpmp_error)
            return None
        """
        np.array([SPKP, KPLP, ...]) stacks the 10 corridor arrays (each 87,600 elements long) as rows, creating a (10 × 87600) matrix. .transpose() flips it to (87600 × 10).
        The final shape of TDC is (87600 × 10) — rows are timesteps, columns are corridors, values are MW.
        The column order is fixed: SPKP=0, KPLP=1, LPGP=2, GPBP=3, BPMP=4, EPMP=5, TISP=6, GILP=7, MIMP=8, KIEP=9. This order must match TDistances in Input.py 
        """
        TDC = np.array([SPKP, KPLP, LPGP, GPBP, BPMP, EPMP, TISP, GILP, MIMP, KIEP ]).transpose() # TDC(t, k), MW
    else:
        TDC = np.zeros((intervals, len(solution.TLoss))) # If len(coverage) == 1 (only one node active, no inter-provincial transmission possible), TDC is a zero matrix of the correct shape.
    
    """
    MStoragePH — the only new calculation in this block. solution.StoragePH is the 87,600-element national PHES storage level time series (MWh). 
    Tiling it to (87600 × 11) and multiplying by pcfactor allocates storage levels to provinces proportionally to their PHES power capacity. 
    This gives MStoragePH[t,j] = MWh of storage attributed to province j at hour t, in MWh.

    """
    if output:
        MStoragePH = np.tile(solution.StoragePH, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        MStorageB = np.tile(solution.StorageB, (nodes, 1)).transpose() * bfactor
        solution.MPV, solution.MIndia, solution.MBaseload, solution.MPeaking = (MPV, MIndia, MBaseload, MPeaking)
        solution.MDischargePH, solution.MChargePH, solution.MStoragePH = (MDischargePH, MChargePH, MStoragePH)
        solution.MDischargeB, solution.MChargeB, solution.MStorageB = (MDischargeB, MChargeB, MStorageB)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)
        #solution.MBaseload_exp, solution.MPV_exp, solution.MPeaking_exp = (MBaseload_exp, MPV_exp, MPeaking_exp)

    """
    Returns the (87600 × 10) TDC matrix to the caller. 
    In Optimisation.py, the caller immediately computes CAC = np.amax(abs(TDC), axis=0) * 1e-3 — 
    the peak flow on each corridor in GW — and uses it in the LCOE cost formula.
    """
    return TDC
