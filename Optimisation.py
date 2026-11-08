# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au
# Modified by: Prabesh Timilsina


from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv
import os

parser = ArgumentParser()
parser.add_argument('-i', default=500, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=2, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-e', default='15', type=str, required=False, help='per-capita electricity = 2_2030, 5_2040, 15, 45, etc.')
parser.add_argument('-n', default='Super', type=str, required=False, help='Super, SP, KP...')
parser.add_argument('-s', default='construction', type=str, required=False, help='existing,construction')
parser.add_argument('-y', default='import', type=str, required=False, help='import, no_import')
parser.add_argument('-q', '--import-penalty-multiplier', default=3.0, type=float, required=False, help='soft-cap penalty multiplier on import energy above cap')
parser.add_argument('-x', default='export', type=str, required=False, help='export, no_export')
parser.add_argument('-b', default='True', type=str, required=False, help='Use fixed contingency-sized battery=True, PV+PHES only=False')
parser.add_argument('--gross-import-cap-fraction', default=0.2, type=float, required=False, help='import cap threshold (0-1): controls both the hard physical cap and the soft penalty threshold')
parser.add_argument('--gross-export-cap-fraction', default=2.0, type=float, required=False, help='export cap threshold (0-1): controls both the hard physical cap and the soft penalty threshold')
parser.add_argument('--export-penalty-multiplier', default=2.0, type=float, required=False, help='soft-cap penalty multiplier on export energy above cap')
#parser.add_argument('-f', default='HVAC', type=str, required=False, help='ac_flag, no_import')
args = parser.parse_args()

# These four lines promote the parsed arguments into module-level variables. 
scenario = args.s
node = args.n
percapita = args.e
#ac_flag = args.f
import_flag = (args.y == 'import')
import_penalty_multiplier = args.import_penalty_multiplier
export_flag = (args.x == 'export')
gross_import_cap_fraction = max(0.0, min(1.0, args.gross_import_cap_fraction))
gross_export_cap_fraction = max(0.0, min(1.0, args.gross_export_cap_fraction))
export_penalty_multiplier = args.export_penalty_multiplier

if args.b == "True":
    batteryScenario = True
elif args.b == "False":
    batteryScenario = False
else:
    print("-b must be True or False")
    exit()


from Input import * 
from Simulation import Reliability 
from Network import Transmission

factor_dict = dict(np.genfromtxt('Data/factor_hvac.csv', dtype=None, delimiter=',', encoding=None))
transmission_cost = np.array([factor_dict['SPKP'], factor_dict['KPLP'], factor_dict['LPGP'], factor_dict['GPBP'], factor_dict['BPMP'], factor_dict['EPMP'], factor_dict['TISP'], factor_dict['GILP'], factor_dict['MIMP'], factor_dict['KIEP']])

def apply_gross_import_cap(india_imports, gross_cap_fraction, annual_demand_mwh, years_count):
    """Enforce a hard cap on gross imports: for each year, if total positive
    (import) energy exceeds gross_cap_fraction * that year's demand, uniformly
    scale down all positive import hours so that the gross import total equals
    exactly the allowed budget.  Export (negative) hours are untouched.
    
    Parameters
    ----------
    india_imports : 1-D array (length = years_count * 8760), MW
    gross_cap_fraction : float in [0, 1]
    annual_demand_mwh : float, total average annual demand in MWh
    years_count : int, number of years in the simulation
    
    Returns
    -------
    india_imports : modified in-place and returned
    """
    for yr in range(years_count):
        start = yr * 8760
        end = (yr + 1) * 8760
        year_slice = india_imports[start:end]

        # Gross imports = sum of all positive hours (MW * 1 hour resolution = MWh)
        positive_mask = year_slice > 0
        gross_import_mwh = year_slice[positive_mask].sum() * 1.0  # resolution = 1 hour per timestep (MW * 1 h = MWh)
        budget_mwh = gross_cap_fraction * annual_demand_mwh

        if gross_import_mwh > budget_mwh and gross_import_mwh > 0:
            scale = budget_mwh / gross_import_mwh
            year_slice[positive_mask] *= scale

    return india_imports

def apply_gross_export_cap(india_imports, gross_cap_fraction, annual_demand_mwh, years_count):
    """Enforce a hard cap on gross exports: for each year, if total negative
    (export) energy (in absolute terms) exceeds gross_cap_fraction * that year's demand,
    uniformly scale down all negative export hours so that the gross export total equals
    exactly the allowed budget. Import (positive) hours are untouched.
    
    Parameters
    ----------
    india_imports : 1-D array (length = years_count * 8760), MW
    gross_cap_fraction : float in [0, 1]
    annual_demand_mwh : float, total average annual demand in MWh
    years_count : int, number of years in the simulation
    
    Returns
    -------
    india_imports : modified in-place and returned
    """
    for yr in range(years_count):
        start = yr * 8760
        end = (yr + 1) * 8760
        year_slice = india_imports[start:end]

        # Gross exports = sum of all negative hours (absolute value)
        negative_mask = year_slice < 0
        gross_export_mwh = -year_slice[negative_mask].sum() * 1.0
        budget_mwh = gross_cap_fraction * annual_demand_mwh

        if gross_export_mwh > budget_mwh and gross_export_mwh > 0:
            scale = budget_mwh / gross_export_mwh
            year_slice[negative_mask] *= scale

    return india_imports

def F(x):
    """This is the objective function.
        F is defined as a Python function that takes one argument: x, 
        the decision variable vector. It returns one number: the objective value (LCOE + penalties). 
        The DE algorithm will call this function
    """

    # Initialise the optimisation
    """
    create a Solution object. x is the raw 39-element NumPy array from the DE algorithm.
    Solution(x) (defined in Input.py) slices x into named fields (S.CPV, S.CPHP, S.CPHS, S.CInter), 
    computes S.GPV (hourly solar generation for all sites), and attaches all module-level data arrays.
    """
    S = Solution(x)

    CIndia = np.nan_to_num(np.array(S.CInter)) # replaces any NaN or Inf values with zero.

    if import_flag == True:
        # Simulation with only baseload
        """
        Simulation 1: run with zero imports
        To discover what the system's natural deficit looks like using only its own domestic resources 
        (solar PV, RoR hydro, peaking hydro, PHES). The deficit that results tells us how large the India links must be to cover it. 
        Without this baseline simulation, there is no way to know what import capacity is actually needed.
        """


        """
        Deficit_energy1 — energy-limited deficit hours (PHES ran empty), MW at each timestep
        Deficit_power1 — power-limited deficit hours (PHES hit power cap), MW at each timestep
        Deficit1 — total deficit (sum of the two above), MW at each timestep — the primary output used below
        DischargePH1 — PHES discharge profile, MW at each timestep
        DischargePeaking1 — peaking hydro discharge profile, MW at each timestep
        Spillage1 — surplus generation that could not be stored, MW at each timestep
        """
        Deficit_energy1, Deficit_power1, Deficit1, DischargePH1, DischargePeaking1, Spillage1 = Reliability(S, baseload=baseload, india_imports=np.zeros(intervals), daily_peaking=daily_peaking, peaking_hours=peaking_hours)
        Max_deficit1 = np.reshape(Deficit1, (-1, 8760)).sum(axis=-1) # MWh per year, penalty and sizing logic below needs to know the worst year, 10 rows (one per year), 8760 columns (one per hour
        PIndia = Deficit1.max() * pow(10, -3) # GW, finds the single largest MW value across all 87,600 hours of Deficit

        GIndia = resolution * (Max_deficit1).max() #takes the maximum of the 10 annual deficit values — the worst year's total deficit energy in MWh. 

        """
        CIndia.sum() is the total import capacity the optimiser has chosen across all four India links (TI + GI + MI + KI), in GW.
        PIndia is the peak power the system actually needs from India (from line 46), in GW.
        Only penalise UNDERSIZED interconnectors (not enough capacity for imports).
        Do NOT penalise oversized interconnectors — extra capacity enables exports
        which can reduce spillage.
        """
        # Penalise only under-sized interconnectors (your original version)
        PenPower = max(0, PIndia - CIndia.sum()) * pow(10,3)
        
        """
        there is no annual energy limit on India imports — Nepal can import as much energy as the line capacity allows, every hour, all year. 
        So the energy penalty is always zero.
        """
        PenEnergy = 0
        
        # Simulation with baseload, all existing capacity
        """
        The second Reliability call. np.ones(intervals) * CIndia.sum() * pow(10,3) creates a flat array of 87,600 values,
        all equal to the total India import capacity in MW — running imports at maximum every single hour for the full 10 years.

        even with the most favourable possible import scenario (full imports 24/7), is there still a deficit? 
        If yes, the candidate solution genuinely lacks domestic capacity — not enough PV, not enough PHES — and deserves a reliability penalty.
        """
        Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=np.ones(intervals) * CIndia.sum() * pow(10,3), daily_peaking=daily_peaking, peaking_hours=peaking_hours)

        # Deficit penalty function
        """
        Deficit.sum() adds up all 87,600 hourly deficit values from Simulation 2 (maximum constant imports). 
        This gives the total energy in MWh that remains unmet over 10 years even with full imports.
        - S.allowance subtracts the permitted tolerance (0.002% of worst-year demand, in MWh). If deficit is within allowance, the result is negative and max(0, ...) returns zero — no penalty. 
        If deficit exceeds allowance, the result is positive — a penalty proportional to the excess.
        
        """
        PenDeficit = max(0, Deficit.sum() * resolution - S.allowance)

        # India import profile
        """
        realistic India import time series
        Deficit1 (from Simulation 1, zero imports) is the deficit profile in MW
        clamps this to the range [0, total_import_capacity_in_MW]
        import exactly as much as needed (the deficit value) but never more than the line can carry (clipped to capacity).

        The lower clip of 0 ensures no negative imports — the model does not allow Nepal to export to India here.
        """
        # Determine clipping bounds based on scenario flags in Input.py
        low = -CIndia.sum() * pow(10,3) if allow_exports else 0
        high = CIndia.sum() * pow(10,3) if allow_imports else 0
        # Allow negative values for export when permitted; otherwise constrain to allowed directions
        india_imports = np.clip(Deficit1 - Spillage1, low, high) # MW

        # Hard cap: limit gross imports to gross_import_cap_fraction of annual demand
        annual_demand_mwh = MLoad.sum() * resolution / years  # MWh per year

        # Measure uncapped gross import energy BEFORE applying the hard cap,
        # so the soft penalty can see how much the optimizer wanted to over-import.
        uncapped_gross_import_mwh = india_imports[india_imports > 0].sum() * 1.0  # MW * 1h = MWh (total over all years)

        # Measure uncapped gross export energy BEFORE applying the hard cap,
        # so the soft penalty can see how much the optimizer wanted to over-export.
        uncapped_gross_export_mwh = -india_imports[india_imports < 0].sum() * 1.0  # MW * 1h = MWh (total over all years)

        india_imports = apply_gross_import_cap(india_imports, gross_import_cap_fraction, annual_demand_mwh, years)
        india_imports = apply_gross_export_cap(india_imports, gross_export_cap_fraction, annual_demand_mwh, years)

        # Simulation using the existing capacity generation profiles - required for storage average annual discharge
        """
        Uses the india_imports profile- realistic, demand-responsive imports capped at line capacity.
        This simulation produces the physically meaningful results that feed directly into the LCOE calculation.

        The results again overwrite the un-suffixed variables. 
        After this line, Deficit, DischargePH, DischargePeaking, and Spillage all reflect the system state under realistic operation
        these are the values stored on S and later used by Network.py and Statistics.py.
        """
        Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=india_imports, daily_peaking=daily_peaking, peaking_hours=peaking_hours)

        # Charged energy into storage systems for charging-related variable costs.
        GPHES = S.ChargePH.sum() * resolution / years * pow(10,-6) # TWh per year
        GBattery = S.ChargeB.sum() * resolution / years * pow(10,-6)

        """
        The no_import path is simpler — just one simulation, no import sizing
        PenPower = 0 and PenEnergy = 0 — there are no import links to size, so both import-related penalties are zero.
        Single Reliability call — runs the simulation with zero imports. Whatever deficit remains must be covered by more domestic PV and PHES
        """
    else:
        PenPower = 0
        PenEnergy = 0
        uncapped_gross_import_mwh = 0
        uncapped_gross_export_mwh = 0
        annual_demand_mwh = 0

        india_imports = np.zeros(intervals)

        # Simulation using the existing capacity generation profiles - required for storage average annual discharge
        Deficit_energy, Deficit_power, Deficit, DischargePH, DischargePeaking, Spillage = Reliability(S, baseload=baseload, india_imports=india_imports, daily_peaking=daily_peaking, peaking_hours=peaking_hours)

        # Deficit penalty function
        PenDeficit = max(0, Deficit.sum() * resolution - S.allowance)

        # Charged energy into storage systems for charging-related variable costs.
        GPHES = S.ChargePH.sum() * resolution / years * pow(10,-6) # TWh per year
        GBattery = S.ChargeB.sum() * resolution / years * pow(10,-6)

    # Transmission capacity calculations
    """
    If 'Super' in node is True (which it is when node == 'Super'), call Transmission(S, domestic_only=True, output=True). Otherwise, fill TDC with zeros.
    domestic_only=True — use only provincial net import/export without any export flows.
    output=True — instructs Transmission to write all disaggregated provincial arrays (MPV, MIndia, MBaseload, etc.) back onto S
    """
    # TDC is an (87,600 × 10) array — 10 corridor flows at each of 87,600 hours.
    TDC = Transmission(S, domestic_only=True, output=True) if 'Super' in node else np.zeros((intervals, len(TLoss)))
    if TDC is None:
        return np.inf

    # abs because for capacity planning only the magnitude matters.
    """
    finds the maximum absolute flow on each of the 10 corridors across all 87,600 hours.
    This gives the peak power that ever flowed on each corridor — the minimum transmission capacity that must be built on that line to accommodate the worst-case hour.
    CAC is a 10-element array in GW — the peak capacity required on each of the 10 corridors
    """
    CAC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW

    # Transmission penalty function\
    # no hard transmission capacity limits imposed during optimisation 
    PenDC = 0

    # Average annual electricity generated by existing capacity
    # all hours of RoR hydro generation across all provinces and all 87,600 hours + all peaking hydro dispatch hours
    # /efficiencyPH. This is an accounting convention that ensures the cost calculation is internally consistent with how the PPA cost factor was defined in factor_hvac.csv.
    # / years converts the 10-year total to an annual average.
    # GHydro is a single scalar in MWh/year representing the average annual electricity generated by all hydro (RoR + peaking). Used in the cost array
    GHydro = resolution * (baseload.sum() + DischargePeaking.sum()) / efficiencyPH / years
    
    # Average annual electricity exchanged through external interconnections
    # Split into gross imports and gross exports for separate cost factors
    GIndia_imp = resolution * india_imports[india_imports > 0].sum() / years  # gross import MWh/year (positive)
    GIndia_exp = resolution * (-india_imports[india_imports < 0].sum()) / years  # gross export MWh/year (positive value)
    GIndia = resolution * india_imports.sum() / years  # net exchange MWh/year (kept for penalty/legacy use)

    # Soft penalty on import over-reliance: uses the UNCAPPED gross import energy
    # (measured before the hard cap was applied) so the optimizer is penalised for
    # solutions that would have exceeded the import cap.  This steers the DE toward
    # building more domestic PV/PHES rather than silently relying on the hard cap.
    if import_flag:
        budget_mwh_total = gross_import_cap_fraction * annual_demand_mwh * years  # total allowed over all years
        import_excess_mwh = max(0, uncapped_gross_import_mwh - budget_mwh_total)
        import_excess_twh = import_excess_mwh * pow(10, -6)  # MWh → TWh
        if import_excess_twh > 0:
            PenEnergy += import_penalty_multiplier * factor_dict['India-imp'] * import_excess_twh

        if export_flag:
            export_budget_mwh_total = gross_export_cap_fraction * annual_demand_mwh * years
            export_excess_mwh = max(0, uncapped_gross_export_mwh - export_budget_mwh_total)
            export_excess_twh = export_excess_mwh * pow(10, -6)  # MWh → TWh
            if export_excess_twh > 0:
                PenEnergy += export_penalty_multiplier * factor_dict['India-exp'] * export_excess_twh

    # Levelised cost of electricity calculation
    """
    factor is the array of annualised cost multipliers loaded from factor_hvac.csv
    The array in brackets is the "quantity" vector — each element represents a capacity (in GW) or energy (in TWh) of one technology. 
    The element-wise product factor * quantities gives the annual cost for each technology. 
    Then .sum() adds them all up to get the total annual system cost in $b/year.
    list(CAC) unpacks the 10 corridor capacity values into 10 separate positions in the array.
    """

    """
    The cost array includes:
    Total solar PV installed capacity
    Wind (not used, placeholder)
    Annual India import energy
    Total PHES power capacity
    PHES energy storage capacity
    Annual PHES charging energy
    Peak flow on each of 10 corridors
    PV connection infrastructure cost
    Wind connection (not used)
    Annual hydro generation (PPA cost)
    """
    # Battery is now a fixed exogenous grid stability reserve with fixed cost
    # (not optimized as a decision variable)
    battery_fixed_cost = factor_dict['BP'] * sum(S.CBP) + factor_dict['BS'] * S.CBS
    
    cost = (
        factor_dict['PV'] * sum(S.CPV)
        + factor_dict['India-imp'] * GIndia_imp * pow(10, -6)  # cost of importing from India
        - factor_dict['India-exp'] * GIndia_exp * pow(10, -6)  # revenue from exporting to India (reduces cost)
        + factor_dict['PHP'] * sum(S.CPHP)
        + factor_dict['PHS'] * S.CPHS
        + factor_dict['PHES-Charging'] * GPHES
        + factor_dict['B-Charging'] * GBattery
        + (transmission_cost * CAC).sum()
        + factor_dict['ACPV'] * sum(S.CPV)
        + factor_dict['Hydro'] * GHydro * pow(10, -6)
        + battery_fixed_cost  # Fixed battery cost added directly
    ) # $b p.a.

    """
    Transmission losses per corridor:
    abs(TDC) gives the absolute MW flow on each corridor at each hour (87,600 × 10). 
    .sum(axis=0) sums down all timesteps, giving the cumulative MWh transmitted on each of the 10 corridors over 10 years. 
    Multiplying element-wise by TLoss (the fraction lost per corridor) gives the energy lost in wires on each corridor.
    """
    loss = np.sum(abs(TDC), axis=0) * TLoss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss) 


    # PV capacity penalty (Max 180 GW cap across the country)
    # PV capped to 20% of Hydro capacity
    # 6641 = 1.33
    # 10000 = 2
    #13500 = 2.70
    #16143 = 3.23
    PenPV = max(0, sum(S.CPV) - 2) * pow(10, 4)
    PenPHES = max(0, S.CPHS - 500) * pow(5, 2)

    # Spillage hard cap: penalise solutions where average annual spillage exceeds threshold
    # Change 5.0 to whatever TWh/year limit you want
    annual_spillage_twh = Spillage.sum() * resolution / years * pow(10, -6)  # TWh per year
    PenSpillage = max(0, annual_spillage_twh - 3.0) * 10000  # large multiplier → near-hard cap

    if not os.path.exists('Results'):
        os.makedirs('Results')

    with open('Results/record_{}_{}_{}_{}_{}.csv'.format(node, scenario, percapita, import_flag, batteryScenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Append battery sizing (power in GW, energy in GWh) to the results for reference
        writer.writerow(np.append(x,[PenDeficit+PenEnergy+PenPower+PenDC+PenSpillage+PenPHES,PenDeficit,PenEnergy,PenPower,LCOE,battery_fr_power_gw,battery_fr_energy_gwh]))

    """
    PenDeficit — pushes the DE away from solutions that leave demand unmet. If large, the DE will favour solutions with more PV and PHES even at higher cost
    PenEnergy — always 0 in Nepal. Would penalise over-use of import energy.
    PenPower — pushes the DE toward correctly sizing the India import links. If large, the DE will adjust CInter toward the value that matches PIndia.
    PenEnergy, PenDC — always 0 in Nepal
    """
    Func = LCOE + PenDeficit + PenEnergy + PenPower + PenDC + PenPV + PenSpillage + PenPHES
    
    return Func

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    """
    pv_lb, Input.py, 23(length), 0.001 GW per PV site 
    contingency_ph, Input.py, 11, 25% of peak load per node (GW)
    [0.] literal 1 PHES energy ≥ 0 GWh
    [0.] * inters, literal × 4, 4, India link capacity ≥ 0 GW

    Battery has been removed from decision variables — it is now a fixed exogenous constant.
    Decision vector x now has 39 dimensions (previously 51).
    The DE algorithm enforces these bounds hard — it never proposes an x value outside [lb[i], ub[i]] for any dimension i
    """

    lb = pv_lb + contingency_ph + [0.] + [0.] * inters
    ub = pv_ub + phes_ub + phes_s_ub + inters_ub

    """
    func=F — the objective function. DE will call F(x) for every candidate solution, every generation.
    bounds=list(zip(lb, ub)) — zip(lb, ub) pairs each lower bound with its corresponding upper bound
    tol=0 — tolerance for convergence. tol=0 means "never stop early based on apparent convergence." 
    The algorithm runs for exactly maxiter generations regardless of whether the objective has stopped improving.

    updating='deferred' — all population members are evaluated before any update is applied. This is required for workers=-1 (parallel evaluation).
    """
    result = differential_evolution(func=F, bounds=list(zip(lb, ub)), tol=0, # init=start,
                                    maxiter=args.i, popsize=args.p, mutation=args.m, recombination=args.r,
                                    disp=True, polish=False, updating='deferred', workers=-1) ###### CHANGE WORKERS BACK TO -1

    with open('Results/Optimisation_resultx_{}_{}_{}_{}_{}.csv'.format(node,scenario,percapita,import_flag,batteryScenario), 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    """
    Analysis(result.x, '_Super_existing_2_True.csv') — calls Fill.py's main function with two arguments:

    result.x — the 39 optimal capacity values, passed directly from the DE result
    '_Super_existing_2_True.csv' — the suffix string used to name all output files

    Analysis will run the backward-looking deficit fill algorithm, produce the realistic India import schedule,
    and then call Statistics.py which produces the final LPGM and GGTA CSV outputs.
    """
    from Fill import Analysis
    Analysis(result.x,'_{}_{}_{}_{}_{}.csv'.format(node,scenario,percapita,import_flag,batteryScenario))
