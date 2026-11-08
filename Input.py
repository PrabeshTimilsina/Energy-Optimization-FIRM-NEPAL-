# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au
# Modified by: Prabesh Timilsina


import numpy as np
from Optimisation import scenario, node, percapita, import_flag, export_flag, batteryScenario

###### NODAL LISTS ######

Nodel = np.array(['SP', 'KP', 'LP', 'GP', 'BP', 'MP', 'EP', 'TI','GI', 'MI', 'KI'])
# SP has 3 PV sites, so SP *3, similarly for others
PVl = np.array(['SP'] * 3 + ['KP'] * 3 + ['LP'] * 2 + ['GP'] * 3 + ['BP'] * 3 + ['MP'] * 3 + ['EP'] * 6)
# 3 sites in 'SP' have upper bound of 22, 28, 15 MW, similarly for others. For nodes without PV sites, the upper bound is set to 0.
pv_ub_np = np.array([12., 12., 12.] + [12., 12., 12.] + [10., 10.] + [10., 10., 10.] + [10., 10., 12.] + [12., 10., 10.] + [12., 12., 12., 10., 12., 10.])
#the upper bound of phes sites, maximum power capacity installable at each internal and external nodes, cant build PHES in India so 0 for last 4 values.
phes_ub_np = np.array([20.] + [20.] + [10.] + [20.] + [20.] + [5.] + [20.] + [0.] + [0.] + [0.] + [0.])

# Add external interconnections 
Interl = np.array(['TI']*1 + ['GI']*1 + ['MI']*1 + ['KI']*1) if node == 'Super' else np.array([])
#hours per timestamp, we have one, 30 min resolution is 0.5
resolution = 1

###### DATA IMPORTS ######
#electricity demand of one proveince/node in MW per hour. 87600*11
MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW
#normalized capacity factor of solar PV, 87600*23, value between 0 and 1, 0 for night, high for day.
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), dimensionless

#skip 1 to 3 columns, which are node, code, fraction, then Hydro, Hydro ROR, hydro peaking
assets = np.genfromtxt('Data/assets_{}.csv'.format(scenario), dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
#skip 1 to 3, and record constraint
constraints = np.genfromtxt('Data/constraints_{}.csv'.format(scenario), dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)

if scenario == 'existing':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)

elif scenario == '6641':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == '10000':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == '13500':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == 'construction':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == 'midconstruction':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == 'all':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)
elif scenario == 'licence':
    hydrol = np.array(['SP']*1+['KP']*1+['LP']*1+['GP']*1+['BP']*1+['MP']*1+['EP']*1)    

# electricity generation through assets
#unpacks three columns from assets simultaneously.
#CHydro_max: maximum power the province's hydro plants could ever produce simultaneously.
#CHydro_RoR:  it generates whenever the river flows, you cannot hold it back.
#CHydro_Peaking: The portion that can be stored in a reservoir and dispatched on demand.
CHydro_max, CHydro_RoR_raw, CHydro_Peaking_raw = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW

# Disable peaking entirely: convert all peaking capacity to RoR and set peaking capacity to 0
CHydro_RoR = CHydro_max.copy()
CHydro_Peaking = np.zeros_like(CHydro_max)

#the maximum annual energy generation from hydro for each province, This is the physical water resource limit, not a policy limit.
EHydro = constraints[:, 0] # GWh per year
#Each value is the run-of-river hydro generation at that node at that hour in MW
hydroProfiles = np.genfromtxt('Data/RoR_{}.csv'.format(scenario), delimiter=',', skip_header=1, usecols=range(4,4+len(Nodel)), encoding=None).astype(float)

# Define constants
peaking_hours = 4
peaking_start = 18
peaking_end = peaking_start + peaking_hours

# Create base arrays (peaking is completely disabled)
# Baseload is capped by installed hydro generator capacity (physical limit)
baseload = np.minimum(hydroProfiles, CHydro_RoR * 1e3)  # MW
daily_peaking = np.zeros_like(hydroProfiles)  # All zeros since peaking is disabled
 
         
###### CONSTRAINTS ######
# Energy constraints
Hydromax = EHydro.sum() * pow(10,3) # GWh to MWh per year

# Transmission constraints
""" externalImports = 0.05 if node=='Super' else 0
CDC7max, CDC8max, CDC9max, CDC10max = 4 * [externalImports * MLoad.sum() / MLoad.shape[0] / 1000] # 5%: External interconnections: THKD, INSE, PHSB, MW to GW """

###### TRANSMISSION LOSSES ######
# HVAC backbone scenario
ac_flags = np.array([True, True, True, True, True, True, True, True, True, True])
#else:
    # HVAC backbone scenario
#    dc_flags = np.array([False, False, False, False, False, False, True, True, True, True])

TLoss = []
# Distance in KM, one per transmission corridor. ['SPKP', 'KPLP', 'LPGP', 'GPBP', 'BPMP', 'EPMP', 'TISP', 'GILP', 'MIMP', 'KIEP']
# Distances updated to match reference model values (Geeta's model)
TDistances = [105, 205, 86, 101, 108, 124, 140, 125, 130, 110]
TDistances = np.array(TDistances)
#For HVAC (ac_flags[i] = True): loss = distance × 0.07 — HVAC loses 7% per 1000 km. So 131 km loses 131 × 0.07 = 9.17% of transmitted energy.
for i in range(0,len(ac_flags)):
    TLoss.append(TDistances[i]*0.07) if ac_flags[i] else TLoss.append(TDistances[i]*0.03)
#This converts the percentage-per-1000km calculation into a pure fraction
TLoss = np.array(TLoss)* pow(10, -3)
print(TLoss)
    
###### STORAGE SYSTEM CONSTANTS ######
efficiencyPH = 0.8
efficiencyB = 0.9

###### COST FACTORS ######
factor = np.genfromtxt('Data/factor_hvac.csv', delimiter=',', usecols=1)

###### EXPORT / IMPORT FLAGS ######
# Control whether hourly exports/imports are allowed at runtime (research scenarios)
# - `allow_imports`: when True the model may place positive hourly imports up to line capacity
# - `allow_exports`: when True the model may place negative hourly imports (exports) up to line capacity
allow_imports = import_flag
allow_exports = export_flag

###### SIMULATION PERIOD ######
firstyear, finalyear, timestep = (2013, 2022, 1)

###### SCENARIO ADJUSTMENTS #######
# Node values
# coverage is set to all 11 nodes. No data needs to be filtered.
if 'Super' == node:
    coverage = Nodel
# coverage is restricted to a subset of nodes
else:
    if 'APG_PMY_Only' == node:
        coverage = np.array(['SP', 'KP', 'LP', 'GP', 'BP', 'MP', 'EP'])
    elif 'APG_BMY_Only' == node:
        coverage = np.array(['TI','GI', 'MI', 'KI'])
    else:
        print("Undefined network structure. Check value of -n command line argument.")
        exit()
    
    #np.in1d(PVl, coverage) returns a boolean array: True for each solar site whose province is in 
    #pv_ub_np[those_indices] keeps only the upper bounds for active sites.
    pv_ub_np = pv_ub_np[np.where(np.in1d(PVl, coverage)==True)[0]]
    phes_ub_np = phes_ub_np[np.where(np.in1d(Nodel, coverage)==True)[0]]

    # The list comprehension applies the same node-filter to Nodel, PVl, and Interl simultaneously.
    Nodel, PVl, Interl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Interl)]
    
    factor = np.genfromtxt('Data/factor_hvac.csv', delimiter=',', usecols=1)

###### DECISION VARIABLE LIST INDEXES ######
intervals, nodes = MLoad.shape # unpacks the shape of MLoad. `intervals = 87600`, `nodes = 11` (or fewer for sub-networks).
years = int(resolution * intervals / 8760)
pzones = TSPV.shape[1] # Number of Solar PV and wind sites
pidx, phidx = (pzones, pzones + nodes) # PV, PHES power index boundaries.
inters = len(Interl) # Number of external interconnections
iidx = phidx + 1 + inters # Index after PHES energy and external interconnections.
"""
The layout of `x` is therefore (after battery refactoring):
x[0:23]     → CPV[0..22]      (23 PV site capacities, GW)
x[23:34]    → CPHP[0..10]     (11 PHES power values, GW)
x[34]       → CPHS             (1 PHES energy value, GWh)
x[35:39]    → CInter[0..3]    (4 India link capacities, GW)

Battery is completely removed from the decision vector to avoid wasting DE evaluations,
and is instead treated as a fixed exogenous constant. Decision vector now has 39 dimensions
instead of 51, improving optimizer efficiency.
"""

###### NETWORK CONSTRAINTS ######
"""
MLoad.sum() adds up all 87,600 X 11 values — the total MWh of electricity demand across all provinces and all hours over 10 years.
Multiplying by resolution=1 does nothing here (would matter at 0.5-hour resolution).
Dividing by years=10 gives the average annual total. Multiplying by pow(10,-9) = 1e-9 converts MWh to PWh (petawatt-hours, 10¹⁵ Wh).
"""
energy = (MLoad).sum() * pow(10, -9) * resolution / years # PWh p.a.
#the lower bound for PHES power capacity, one value per node. even the minimum acceptable solution must have enough PHES to cover 25% of peak provincial demand.
contingency_ph = list(0.05 * (MLoad).max(axis=0) * pow(10, -3))[:(nodes)] # MW to GW, [:(nodes)] slices to exactly nodes elements in case the max calculation returns more.

# Battery for contingency reserve is sized outside the energy balancing
# optimisation. Power is an exogenous largest credible contingency (N-1 standard).
# Duration: 30-minute hold for frequency regulation (literature standard)
battery_contingency_duration_minutes = 30  # Variable - can be adjusted for sensitivity analysis
battery_contingency_duration_hours = battery_contingency_duration_minutes / 60

largest_contingency_gw = 0.5  # 500 MW Fixed (1200 MW for largest contingency in Nepal)
battery_fr_power_gw = largest_contingency_gw
battery_fr_energy_gwh = battery_fr_power_gw * battery_contingency_duration_hours  # Fixed exogenous capacity

# Distribute the fixed contingency battery across domestic Nepal nodes.
battery_fr_power_by_node = nodes * [0.]
if batteryScenario == True:
    domestic_nodes = nodes - inters
    for j in range(domestic_nodes):
        battery_fr_power_by_node[j] = battery_fr_power_gw / domestic_nodes

   
#manage = 0 # weeks
allowance = min(0.00002*np.reshape(MLoad.sum(axis=1), (-1, 8760)).sum(axis=-1)) # Allowable annual deficit of 0.002%, MWh


###### DECISION VARIABLE UPPER BOUNDS ######
#These convert the NumPy arrays of bounds into Python lists (required by SciPy's differential_evolution), and define the remaining bounds
pv_ub = [x for x in pv_ub_np] 
phes_ub = [x for x in phes_ub_np]
phes_s_ub = [2000.] # the upper bound for the single PHES energy variable CPHS: 2000 GWh
# It needs to be ~8K(2x4GW) in 2031 and ~15K(4x4GW) in 2036
inters_ub = [3.75] * inters if node == 'Super' else [0.] * inters # upper bounds for India interconnector capacities: 4 GW per link

###### DECISION VARIABLE LOWER BOUNDS ######
pv_lb = [.001] * pzones # lower bounds for PV capacities: 0.001 GW = 1 MW minimum at each site


class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    The Solution class is the central data container of the entire model.
    Every function in every file receives a Solution object and uses it to read inputs and write outputs

    mutable state container that carries both the decision variables and all simulation outputs.
    This is a key design pattern — instead of returning arrays from functions, results are attached to the Solution object in-place
    """

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution
        self.baseload = baseload
        #self.indiaExportProfiles = indiaExportProfiles
        self.daily_peaking = daily_peaking

        self.CPV = list(x[: pidx]) # CPV(i), GW
       
       # computes the actual hourly solar PV generation in MW for all 23 sites simultaneously.
        self.GPV = TSPV * np.tile(self.CPV, (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
       

        # self.CPHP = [x[phidx]] # CPHP(j), GW
        self.CPHP = list(x[pidx: phidx]) # CPHP(j), GW
        self.CPHS = x[phidx] # S-CPHS(j), GWh
        self.efficiencyPH = efficiencyPH
        self.efficiencyB = efficiencyB
        
        # Battery is now a fixed exogenous parameter, not a decision variable
        # Use module-level constants for battery capacity
        self.CBP = battery_fr_power_by_node # Battery power capacity by node, GW (fixed)
        self.CBS = battery_fr_energy_gwh # Battery energy capacity, GWh (fixed)

        self.CInter = list(x[phidx+1: iidx]) if node == 'Super' else len(Interl)*[0] #CInter(j), GW
        self.GIndia = np.tile(self.CInter, (intervals, 1)) * pow(10,3) # GInter(j, t), GW to MW

        self.Nodel, self.PVl, self.Hydrol = (Nodel, PVl, hydrol)
        self.Interl = Interl
        self.node = node
        self.scenario = scenario
        self.import_flag = import_flag
        self.allowance = allowance
        self.coverage = coverage
        self.TLoss = TLoss

        self.CHydro_RoR = CHydro_RoR
        self.CHydro_Peaking = CHydro_Peaking
        self.CHydro_max = CHydro_max
        

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)
