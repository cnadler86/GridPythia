"""Genetic algorithm for EMS (Energy Management System) optimization.

Simplified version decoupled from the EOS framework. Configuration is passed
directly via a HEMSConfig dataclass instead of global singletons.
"""

import random
import time
from array import array
from dataclasses import dataclass
from typing import Any, Optional

from deap import algorithms, base, creator, tools
from loguru import logger

from src.config import HEMSConfig
from src.optimization.genetic.genomelayout import GenomeLayout
from src.optimization.params import (
    EnergyManagementParameters,
    OptimizationParameters,
)
from src.optimization.simulation import (
    GridSimulation,
    SimulationResult,
)
from src.simulation.devices import InverterMode, SystemTopology
from src.simulation.devices.battery import Battery
from src.simulation.devices.homeappliance import HomeAppliance
from src.simulation.devices.inverterbase import InverterBase, InverterParameters


def _percentile(data: list[float], p: float) -> float:
    """Return the *p*-th percentile (0 ≤ p ≤ 100) of *data* using linear interpolation."""
    if not data:
        return 0.0
    n = len(data)
    sorted_data = sorted(data)
    k = p / 100.0 * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (k - lo) * (sorted_data[hi] - sorted_data[lo])


@dataclass
class GeneticSolution:
    """Optimization solution produced by the genetic algorithm."""

    inverter_plans: list[dict]
    home_appliance_plans: list[dict]
    result: SimulationResult
    start_solution: Optional[list[int]] = None
    fitness_history: Optional[dict] = None


class GeneticOptimization:
    """Genetic algorithm to solve energy management system optimization.

    Unlike the EOS version, this class receives its configuration directly
    via the `config` parameter instead of global singleton mixins.
    """

    def __init__(
        self,
        config: HEMSConfig,
        verbose: bool = False,
        fixed_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.fix_seed: Optional[int] = fixed_seed
        self.fitness_history: dict[str, Any] = {}

        self.inverters: list[InverterBase] = []
        self.home_appliances: list[HomeAppliance] = []
        self.genome_layout: Optional[GenomeLayout] = None
        self.simulation: Optional[GridSimulation] = None
        self._prices_for_init: list[float] = []
        self._pv_for_init: list[float] = []

        self.toolbox = base.Toolbox()

        if self.fix_seed is not None:
            random.seed(self.fix_seed)
        elif logger.level == "DEBUG":
            self.fix_seed = random.randint(1, 100_000_000_000)  # noqa: S311
            random.seed(self.fix_seed)
            logger.debug("GeneticOptimization: using fixed seed {}", self.fix_seed)

    # ------------------------------------------------------------------
    # Individual creation
    # ------------------------------------------------------------------

    def create_individual(self) -> list[int]:
        """Split-genome conservative initialization."""
        layout = self.genome_layout
        prices = self._prices_for_init
        n = self.config.prediction.hours

        if not layout:
            return []
        if not prices:
            return self._create_random_individual()

        expensive = _percentile(prices[:n], 60)
        cheap = _percentile(prices[:n], 20)

        genome: list[int] = []

        for spec in layout.inverter_specs:
            inv = self.inverters[spec.inverter_index]
            modes_list = inv.available_modes

            idx_idle = next((i for i, m in enumerate(modes_list) if m == InverterMode.IDLE), 0)
            idx_discharge = next(
                (i for i, m in enumerate(modes_list) if m == InverterMode.DISCHARGE_ZERO_FEED_IN),
                next(
                    (i for i, m in enumerate(modes_list) if m == InverterMode.DISCHARGE),
                    idx_idle,
                ),
            )
            idx_ac_charge = next(
                (i for i, m in enumerate(modes_list) if m == InverterMode.AC_CHARGE),
                next(
                    (
                        i
                        for i, m in enumerate(modes_list)
                        if m == InverterMode.AC_CHARGE_ZERO_FEED_IN
                    ),
                    -1,
                ),
            )

            can_discharge = idx_discharge != idx_idle
            can_ac_charge = idx_ac_charge >= 0

            mode_genes: list[int] = []
            for hour in range(n):
                price = prices[min(hour, len(prices) - 1)]
                if price >= expensive and can_discharge:
                    mode_genes.append(idx_discharge)
                elif price <= cheap and can_ac_charge:
                    mode_genes.append(idx_ac_charge)
                else:
                    mode_genes.append(idx_idle)
            genome.extend(mode_genes)

            # Use max rate for AC_CHARGE hours so the cheapest slots are used at full power.
            # Mid rate is fine for all other modes.
            max_rate_idx = max(spec.rate_count - 1, 0)
            mid_rate = max(spec.rate_count // 2, 0)
            for hour in range(n):
                if mode_genes[hour] == idx_ac_charge:
                    genome.append(max_rate_idx)
                else:
                    genome.append(mid_rate)

            if spec.discharge_rate_count > 0:
                mid_discharge = spec.discharge_rate_count // 2
                for _ in range(n):
                    genome.append(mid_discharge)

        if layout.home_appliance_slice:
            for appliance in self.home_appliances:
                genome.append(
                    random.randint(appliance.start_earliest, appliance.start_latest)  # noqa: S311
                )

        return creator.Individual(genome)

    def _create_random_individual(self) -> list[int]:
        """Fallback: fully random genome."""
        layout = self.genome_layout
        if layout is None:
            return []
        genome: list[int] = []
        for spec in layout.inverter_specs:
            n = self.config.prediction.hours
            genome += [random.randint(0, spec.mode_count - 1) for _ in range(n)]  # noqa: S311
            genome += [random.randint(0, max(spec.rate_count - 1, 0)) for _ in range(n)]  # noqa: S311
            if spec.discharge_rate_count > 0:
                genome += [  # noqa: S311
                    random.randint(0, spec.discharge_rate_count - 1)  # noqa: S311
                    for _ in range(n)
                ]
        if layout.home_appliance_slice:
            for appliance in self.home_appliances:
                genome.append(
                    random.randint(appliance.start_earliest, appliance.start_latest)  # noqa: S311
                )
        return creator.Individual(genome)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, individual: list[int]) -> tuple:
        """Mutate *individual* in-place using the split mode/rate genome."""
        layout = self.genome_layout
        if layout is None:
            return (individual,)

        for spec in layout.inverter_specs:
            seg_modes = list(individual[spec.mode_slice])
            (mutated_modes,) = tools.mutUniformInt(
                seg_modes, low=0, up=max(spec.mode_count - 1, 0), indpb=0.15
            )
            individual[spec.mode_slice] = mutated_modes

            if spec.rate_count > 1:
                seg_rates = list(individual[spec.rate_slice])
                (mutated_rates,) = tools.mutUniformInt(
                    seg_rates, low=0, up=spec.rate_count - 1, indpb=0.15
                )
                individual[spec.rate_slice] = mutated_rates

            if spec.discharge_rate_count > 1 and spec.discharge_rate_slice is not None:
                seg_dc = list(individual[spec.discharge_rate_slice])
                (mutated_dc,) = tools.mutUniformInt(
                    seg_dc, low=0, up=spec.discharge_rate_count - 1, indpb=0.15
                )
                individual[spec.discharge_rate_slice] = mutated_dc

        if layout.home_appliance_slice:
            for i, _ in enumerate(self.home_appliances):
                pos = layout.home_appliance_slice.start + i
                if pos < len(individual):
                    (mutated_h,) = self.toolbox.mutate_hour([individual[pos]])
                    individual[pos] = mutated_h[0]

        return (individual,)

    # ------------------------------------------------------------------
    # DEAP setup
    # ------------------------------------------------------------------

    def setup_deap_environment(self, start_hour: int) -> None:
        for attr in ["FitnessMin", "Individual"]:
            if attr in creator.__dict__:
                del creator.__dict__[attr]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox.register(
            "mutate_hour",
            tools.mutUniformInt,
            low=start_hour,
            up=23,
            indpb=0.2,
        )
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    def _build_simulation_arrays(
        self,
        individual: list[int],
    ) -> tuple:
        if self.genome_layout is None:
            return {}, {}, None, []

        layout = self.genome_layout
        n = self.config.prediction.hours
        decoded = layout.decode(individual, self.inverters)

        inverter_modes_all: dict[str, array] = {
            inv.device_id: array("i", [int(InverterMode.IDLE)] * n) for inv in self.inverters
        }
        inverter_rates_all: dict[str, array] = {
            inv.device_id: array("f", [1.0] * n) for inv in self.inverters
        }

        for seg_idx, spec in enumerate(layout.inverter_specs):
            inv = self.inverters[spec.inverter_index]
            modes_h = decoded.inverter_modes[seg_idx]
            rates_h = decoded.inverter_ac_rates[seg_idx]
            inverter_modes_all[inv.device_id] = array("i", [int(m) for m in modes_h[:n]])
            inverter_rates_all[inv.device_id] = array("f", [float(r) for r in rates_h[:n]])

        appliance_load: Optional[array] = None
        applied_starts: list[Optional[int]] = list(decoded.home_appliance_starts)

        if self.home_appliances and decoded.home_appliance_starts:
            total = array("f", [0.0] * n)
            applied_starts = []
            for idx, appliance in enumerate(self.home_appliances):
                raw: Optional[int] = (
                    decoded.home_appliance_starts[idx]
                    if idx < len(decoded.home_appliance_starts)
                    else None
                )
                if raw is not None:
                    applied = appliance.set_starting_time(raw, 0)
                    applied_starts.append(applied)
                    for hr in range(n):
                        total[hr] += appliance.get_load_for_hour(hr)
                else:
                    applied_starts.append(None)
            appliance_load = total

        return inverter_modes_all, inverter_rates_all, appliance_load, applied_starts

    def _simulate(self, individual: list[int], start_hour: int) -> Optional[SimulationResult]:
        if self.simulation is None or self.genome_layout is None:
            return None
        inv_modes, inv_rates, appl_load, _ = self._build_simulation_arrays(individual)
        return self.simulation.simulate(
            inverter_modes=inv_modes,
            inverter_ac_rates=inv_rates,
            appliance_load=appl_load,
            start_idx=start_hour,
            dt=getattr(self.config.prediction, "dt_hours", 1.0),
        )

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        individual: list[int],
        parameters: OptimizationParameters,
        start_hour: int,
        worst_case: bool,
    ) -> tuple:
        try:
            result = self._simulate(individual, start_hour)
        except Exception:
            return (100_000.0,)

        if result is None:
            return (100_000.0,)

        score = result.net_balance * (1.0 if worst_case else -1.0)

        feedin_tariff = parameters.ems.einspeiseverguetung_euro_pro_wh
        for h, fi in enumerate(result.feedin_wh_per_dt):
            if fi <= 0.0:
                continue
            import_price = result.electricity_price_per_dt[h]
            tariff_h = (
                feedin_tariff[h]
                if isinstance(feedin_tariff, list) and h < len(feedin_tariff)
                else float(feedin_tariff)
            )
            gap = import_price - tariff_h
            if gap > 0.0:
                score += fi * gap

        if parameters.eauto:
            penalty = self.config.optimization.genetic.penalties.get("ev_soc_miss", 10)
            for inv in self.inverters:
                if inv.battery and inv.topology in (
                    SystemTopology.EV_CHARGE_ONLY,
                    SystemTopology.EV_V2G,
                ):
                    soc = inv.battery.current_soc_percentage()
                    if (
                        soc < parameters.eauto.min_soc_percentage
                        or soc > parameters.eauto.max_soc_percentage
                    ):
                        score += abs(parameters.eauto.min_soc_percentage - soc) * penalty

        # This is a simple way to penalize losses. Deactivate since double penalty.
        # if not worst_case:
        #     for h, loss_wh in enumerate(result.losses_wh_per_dt):
        #         if loss_wh > 0.0:
        #             score += loss_wh * result.electricity_price_per_dt[h]

        individual.extra_data = (result.net_balance, result.total_losses)  # type: ignore[attr-defined]
        return (score,)

    # ------------------------------------------------------------------
    # Evolutionary loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        start_solution: Optional[list] = None,
        ngen: int = 200,
    ) -> tuple:
        individuals = self.config.optimization.genetic.individuals

        population = self.toolbox.population(n=individuals)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", min)
        stats.register("avg", lambda vals: sum(vals) / len(vals) if vals else 0.0)
        stats.register("max", max)

        if start_solution is not None:
            for _ in range(10):
                population.insert(0, creator.Individual(start_solution))

        pop, log = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=100,
            lambda_=150,
            cxpb=0.6,
            mutpb=0.4,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=self.verbose,
        )

        self.fitness_history = {
            "gen": log.select("gen"),
            "avg": log.select("avg"),
            "max": log.select("max"),
            "min": log.select("min"),
        }

        member: dict[str, list[float]] = {"bilanz": [], "verluste": []}
        for ind in pop:
            if hasattr(ind, "extra_data"):
                bilanz, verluste = ind.extra_data
                member["bilanz"].append(bilanz)
                member["verluste"].append(verluste)

        return hof[0], member

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimierung_ems(
        self,
        parameters: OptimizationParameters,
        start_hour: int = 0,
        worst_case: bool = False,
        ngen: Optional[int] = None,
    ) -> GeneticSolution:
        """Perform EMS optimization and return a GeneticSolution."""
        generations = ngen or self.config.optimization.genetic.generations

        pv_map: dict[str, list[float]] = dict(parameters.ems.pv_prognose_wh)

        # Batteries
        akku: Optional[Battery] = None
        if parameters.pv_akku:
            akku = Battery(
                parameters.pv_akku,
                prediction_hours=self.config.prediction.hours,
            )

        eauto_battery: Optional[Battery] = None
        if parameters.eauto:
            eauto_battery = Battery(
                parameters.eauto,
                prediction_hours=self.config.prediction.hours,
            )

        # Inverters
        inverters: list[InverterBase] = []

        if parameters.inverter:
            inv_params = parameters.inverter
            if (inv_params.pv_source is None or inv_params.pv_source not in pv_map) and pv_map:
                # Fix pv_source to match available PV data
                inv_params = InverterParameters(
                    device_id=inv_params.device_id,
                    battery_id=inv_params.battery_id,
                    pv_source=next(iter(pv_map)),
                    max_ac_output_power_w=inv_params.max_ac_output_power_w,
                    max_ac_charge_power_w=inv_params.max_ac_charge_power_w,
                    dc_to_ac_efficiency=inv_params.dc_to_ac_efficiency,
                    ac_to_dc_efficiency=inv_params.ac_to_dc_efficiency,
                    zero_feed_in=inv_params.zero_feed_in,
                    ac_rates=inv_params.ac_rates,
                )
            inverters.append(InverterBase(parameters=inv_params, battery=akku))

        # EV as InverterBase
        if parameters.eauto and eauto_battery:
            max_charge_w = float(parameters.eauto.max_charge_power_w or 0.0)
            max_discharge_w = float(getattr(parameters.eauto, "max_discharge_power_w", None) or 0.0)
            ev_dc_to_ac = (
                float(parameters.eauto.discharging_efficiency)
                if max_discharge_w > 0
                else (parameters.inverter.dc_to_ac_efficiency if parameters.inverter else 0.95)
            )
            ev_inv_params = InverterParameters(
                device_id=f"{parameters.eauto.device_id}_inverter",
                battery_id=parameters.eauto.device_id,
                pv_source=None,
                max_ac_output_power_w=max_discharge_w if max_discharge_w > 0 else max_charge_w,
                max_ac_charge_power_w=max_charge_w,
                dc_to_ac_efficiency=ev_dc_to_ac,
                ac_to_dc_efficiency=float(parameters.eauto.charging_efficiency),
                zero_feed_in=False,
            )
            inverters.append(InverterBase(parameters=ev_inv_params, battery=eauto_battery))

        self.inverters = inverters

        # Home appliances
        dishwasher: Optional[HomeAppliance] = None
        if parameters.dishwasher is not None:
            dishwasher = HomeAppliance(
                parameters=parameters.dishwasher,
                optimization_hours=self.config.optimization.horizon_hours,
                prediction_hours=self.config.prediction.hours,
            )
        self.home_appliances = [dishwasher] if dishwasher else []

        # Genome layout
        self.genome_layout = GenomeLayout(
            inverters=inverters,
            prediction_hours=self.config.prediction.hours,
            home_appliance_count=len(self.home_appliances),
        )

        self._prices_for_init = list(parameters.ems.strompreis_euro_pro_wh)
        self._pv_for_init = [
            sum(pv_map[k][h] for k in pv_map if h < len(pv_map[k]))
            for h in range(self.config.prediction.hours)
        ]

        # Simulation instance
        ems_params = EnergyManagementParameters(
            pv_prognose_wh=pv_map,
            strompreis_euro_pro_wh=parameters.ems.strompreis_euro_pro_wh,
            einspeiseverguetung_euro_pro_wh=parameters.ems.einspeiseverguetung_euro_pro_wh,
            preis_euro_pro_wh_akku=parameters.ems.preis_euro_pro_wh_akku,
            gesamtlast=parameters.ems.gesamtlast,
        )
        self.simulation = GridSimulation(
            parameters=ems_params,
            optimization_hours=self.config.optimization.horizon_hours,
            inverters=inverters,
            home_appliances=self.home_appliances,
        )

        # DEAP setup
        self.setup_deap_environment(start_hour)
        self.toolbox.register(
            "evaluate",
            lambda ind: self.evaluate(ind, parameters, start_hour, worst_case),
        )

        # Run optimization
        t0 = time.time()
        best_individual, extra_data = self.optimize(parameters.start_solution, ngen=generations)
        logger.debug("GeneticOptimization: elapsed {:.4f} s", time.time() - t0)

        # Final simulation
        inv_modes, inv_rates, appl_load, applied_starts = self._build_simulation_arrays(
            best_individual
        )
        final_result = self.simulation.simulate(
            inverter_modes=inv_modes,
            inverter_ac_rates=inv_rates,
            appliance_load=appl_load,
            start_idx=start_hour,
            dt=getattr(self.config.prediction, "dt_hours", 1.0),
        )
        decoded = self.genome_layout.decode(best_individual, self.inverters)

        # InverterPlan list
        inverter_plans: list[dict] = []
        for seg_idx, spec in enumerate(self.genome_layout.inverter_specs):
            inv = self.inverters[spec.inverter_index]
            modes_h = (
                decoded.inverter_modes[seg_idx] if seg_idx < len(decoded.inverter_modes) else []
            )
            rates_h = (
                decoded.inverter_ac_rates[seg_idx]
                if seg_idx < len(decoded.inverter_ac_rates)
                else []
            )
            is_ev_inv = inv.topology in (
                SystemTopology.EV_CHARGE_ONLY,
                SystemTopology.EV_V2G,
            )
            inverter_plans.append(
                {
                    "inverter_id": inv.device_id,
                    "modes": [int(m) for m in modes_h],
                    "rates": [float(r) for r in rates_h],
                    "is_ev": is_ev_inv,
                    "battery_device_id": inv.parameters.battery_id,
                }
            )

        # HomeAppliancePlan list
        home_appliance_plans: list[dict] = []
        for idx, appliance in enumerate(self.home_appliances):
            start = applied_starts[idx] if idx < len(applied_starts) else None
            home_appliance_plans.append(
                {
                    "appliance_id": appliance.parameters.device_id,
                    "start_hour": start,
                }
            )

        return GeneticSolution(
            inverter_plans=inverter_plans,
            home_appliance_plans=home_appliance_plans,
            result=final_result,
            start_solution=[int(x) for x in best_individual],
            fitness_history=self.fitness_history,
        )
