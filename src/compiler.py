from dataclasses import dataclass, replace
from typing import Optional

from toy import ExperimentConfig, run_experiment
from contract_card import build_contract_card, print_contract_card, save_contract_card
from utils import ensure_dir


@dataclass
class Contract:
    coverage_target: float
    beta: float = 1e-3
    max_dp_eps_train: float = 8.0
    max_dp_eps_cal: float = 8.0
    objective: str = "Among configurations that satisfy the contract, pick the least conservative one."


@dataclass
class SearchGrid:
    # TODO: Maybe include targets here?
    dp_eps_train: list[float]
    dp_eps_cal: list[float]
    nominal_coverage: list[float]
    cal_size: list[int]
    seeds: list[int]


@dataclass
class RegionResult:
    formal_checked: list[tuple[ExperimentConfig, dict]]
    formal_feasible: list[tuple[ExperimentConfig, dict]]
    formal_frontier: list[tuple[ExperimentConfig, dict]]
    empirical_checked: list[tuple[ExperimentConfig, dict]]
    empirical_feasible: list[tuple[ExperimentConfig, dict]]
    best_formal_candidate: tuple[ExperimentConfig, dict] | None
    num_formal_evals: int
    num_empirical_evals: int
    grid_size_total: int


@dataclass
class CompileResult:
    status: str  # "FEASIBLE" | "FORMALLY_INFEASIBLE" | "EMPIRICALLY_INFEASIBLE"
    config: Optional[ExperimentConfig]
    formal_result: Optional[dict]
    empirical_result: Optional[dict]
    certificate_result: Optional[dict]
    num_formal_evals: int
    num_empirical_evals: int
    compiler_mode: str
    frontier_rank_rule: str
    grid_size_total: int


def make_base_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="cifar10",
        data_dir="data",
        out_dir="toy_out",
        seed=0,
        device="cuda",
        batch_size=128,
        epochs_np=10,
        epochs_dp=4,
        verbose=False,
        coverage_target=0.8,
        nominal_coverage=0.8,
        dp_eps_train=4.0,
        dp_delta=1e-5,
        dp_eps_cal=4.0,
        num_bins=50,
        beta=1e-3,
        temperature=2.0,
        label_smoothing=0.5,
        cal_size=2000,
        write_report=False,
    )


def candidate_configs(base: ExperimentConfig, contract: Contract, grid: SearchGrid):
    for teps in grid.dp_eps_train:
        for ceps in grid.dp_eps_cal:
            for nomcov in grid.nominal_coverage:
                for csize in grid.cal_size:
                    yield replace(
                        base,
                        dp_eps_train=teps,
                        dp_eps_cal=ceps,
                        nominal_coverage=nomcov,
                        coverage_target=contract.coverage_target,
                        beta=contract.beta,
                        cal_size=csize,
                    )


def formal_check(cfg: ExperimentConfig, contract: Contract) -> dict:
    coverage_lower_bound_formal = cfg.nominal_coverage - cfg.beta
    coverage_bound_formal_ok = coverage_lower_bound_formal >= cfg.coverage_target

    privacy_training_ok = cfg.dp_eps_train <= contract.max_dp_eps_train
    privacy_cal_ok = cfg.dp_eps_cal <= contract.max_dp_eps_cal

    overall_formal_ok = (
        coverage_bound_formal_ok
        and privacy_training_ok
        and privacy_cal_ok
    )

    return {
        "config": {
            "dp_eps_train": cfg.dp_eps_train,
            "dp_eps_cal": cfg.dp_eps_cal,
            "nominal_coverage": cfg.nominal_coverage,
            "coverage_target": cfg.coverage_target,
            "seed": cfg.seed,
        },
        "formal": {
            "overall_formal_ok": overall_formal_ok,
            "coverage_bound_formal_ok": coverage_bound_formal_ok,
            "privacy_training_ok": privacy_training_ok,
            "privacy_cal_ok": privacy_cal_ok,
            "coverage_lower_bound_formal": coverage_lower_bound_formal,
        },
    }


def formal_search(base: ExperimentConfig, contract: Contract, grid: SearchGrid):
    checked: list[tuple[ExperimentConfig, dict]] = []
    feasible: list[tuple[ExperimentConfig, dict]] = []
    n = 0
    formal_seed = grid.seeds[0]

    for cfg in candidate_configs(base, contract, grid):
        cfg = replace(cfg, seed=formal_seed)
        result = formal_check(cfg, contract)
        checked.append((cfg, result))
        n += 1
        if result["formal"]["overall_formal_ok"]:
            feasible.append((cfg, result))

    return checked, feasible, n

def best_formal_candidate(formal_checked: list[tuple[ExperimentConfig, dict]]):
    if not formal_checked:
        return None

    return max(
        formal_checked,
        key=lambda x: (
            x[1]["formal"]["coverage_lower_bound_formal"],
            -x[0].nominal_coverage,
            -x[0].dp_eps_cal,
            -x[0].dp_eps_train,
        ),
    )

def dominates_for_frontier(cfg_a: ExperimentConfig, cfg_b: ExperimentConfig) -> bool:
    """
    Heuristic dominance for choosing least-conservative formal candidates.
    A dominates B if A is at least as aggressive in every search dimension and
    strictly more aggressive in at least one dimension.
    """
    at_least_as_aggressive = (
        cfg_a.nominal_coverage <= cfg_b.nominal_coverage
        and cfg_a.dp_eps_cal >= cfg_b.dp_eps_cal
        and cfg_a.dp_eps_train >= cfg_b.dp_eps_train
    )
    strictly_more_aggressive = (
        cfg_a.nominal_coverage < cfg_b.nominal_coverage
        or cfg_a.dp_eps_cal > cfg_b.dp_eps_cal
        or cfg_a.dp_eps_train > cfg_b.dp_eps_train
    )
    return at_least_as_aggressive and strictly_more_aggressive


def formal_frontier_candidates(formal_feasible: list[tuple[ExperimentConfig, dict]]):
    """
    Keep only formally feasible candidates that are not dominated by a less
    conservative formally feasible point.
    """
    frontier: list[tuple[ExperimentConfig, dict]] = []

    for cfg, fres in formal_feasible:
        dominated = False
        for cfg2, _ in formal_feasible:
            if cfg2 is cfg:
                continue
            if dominates_for_frontier(cfg2, cfg):
                dominated = True
                break
        if not dominated:
            frontier.append((cfg, fres))

    frontier.sort(
        key=lambda x: (x[0].nominal_coverage, -x[0].dp_eps_cal, -x[0].dp_eps_train)
    )
    return frontier

def summarize_frontier_points(points: list[tuple[ExperimentConfig, dict]]) -> list[dict]:
    out = []
    for cfg, result in points:
        formal = result.get("formal", {})
        empirical = result.get("empirical", {})
        out.append({
            "config": {
                "dp_eps_train": cfg.dp_eps_train,
                "dp_eps_cal": cfg.dp_eps_cal,
                "nominal_coverage": cfg.nominal_coverage,
                "coverage_target": cfg.coverage_target,
                "cal_size": cfg.cal_size,
                "seed": cfg.seed,
            },
            "formal": formal,
            "empirical": empirical,
        })
    return out

def is_pathological(result: dict) -> bool:
    tau = result["empirical"]["tau_dp_cal"]
    return tau >= 0.999


def objective(cfg: ExperimentConfig, result: dict):
    """
    Lower is better.
    Primary: smaller prediction sets.
    Secondary: avoid tau \approx 1, prefer smaller nominal coverage,
    larger eps_cal, larger eps_train, then smaller total epsilon.
    """
    avg_size = result["empirical"]["avg_set_size_dpcal"]
    cert_width = float(result["empirical"].get("certificate_width_tau", float("inf")))
    tau_penalty = 100.0 if is_pathological(result) else 0.0
    total_eps = cfg.dp_eps_train + cfg.dp_eps_cal
    return (
        cert_width,
        avg_size + tau_penalty,
        cfg.nominal_coverage,
        -cfg.dp_eps_cal,
        -cfg.dp_eps_train,
        total_eps,
    )


def summarize_seed_results(cfg: ExperimentConfig, seed_results: list[dict], formal_result: dict) -> dict:
    mean_cov = sum(r["empirical"]["test_coverage_dpcal"] for r in seed_results) / len(seed_results)
    mean_size = sum(r["empirical"]["avg_set_size_dpcal"] for r in seed_results) / len(seed_results)
    mean_tau = sum(r["empirical"]["tau_dp_cal"] for r in seed_results) / len(seed_results)
    mean_acc = sum(r["empirical"]["test_accuracy_dp"] for r in seed_results) / len(seed_results)
    mean_cert = sum(r["full_report"]["dp_calibration"]["certificate_width_tau"] for r in seed_results) / len(seed_results)
    mean_obs_grid = sum(r["full_report"]["dp_calibration"]["observed_inflation_tau_grid"] for r in seed_results) / len(seed_results)
    theorem_tau_rate = sum(float(r["full_report"]["dp_calibration"]["theorem_tau_ok"]) for r in seed_results) / len(seed_results)
    theorem_idx_rate = sum(float(r["full_report"]["dp_calibration"]["theorem_idx_ok"]) for r in seed_results) / len(seed_results)

    return {
        "config": {
            "dataset": cfg.dataset,
            "dp_eps_train": cfg.dp_eps_train,
            "dp_eps_cal": cfg.dp_eps_cal,
            "nominal_coverage": cfg.nominal_coverage,
            "coverage_target": cfg.coverage_target,
            "cal_size": cfg.cal_size,
            "label_smoothing": cfg.label_smoothing,
            "temperature": cfg.temperature,
        },
        "formal": formal_result["formal"],
        "certificate": {
            "certificate_width_tau": mean_cert,
            "observed_inflation_tau_grid": mean_obs_grid,
            "theorem_tau_ok_rate": theorem_tau_rate,
            "theorem_idx_ok_rate": theorem_idx_rate,
        },
        "empirical": {
            "overall_empirical_ok": True,
            "coverage_empirical_ok": True,
            "test_accuracy_dp": mean_acc,
            "test_coverage_dpcal": mean_cov,
            "avg_set_size_dpcal": mean_size,
            "tau_dp_cal": mean_tau,
        },
    }


def empirical_refine(frontier: list[tuple[ExperimentConfig, dict]], seeds: list[int]):
    """
    Evaluate frontier configs empirically. Return:
      - all checked summaries
      - feasible summaries
      - eval count
    """
    n = 0
    checked: list[tuple[ExperimentConfig, dict]] = []
    feasible: list[tuple[ExperimentConfig, dict]] = []

    for cfg, formal_result in frontier:
        seed_results = []
        all_pass = True

        for seed in seeds:
            cfg_seed = replace(cfg, seed=seed)
            result = run_experiment(cfg_seed)
            n += 1
            seed_results.append(result)

            if not result["empirical"]["overall_empirical_ok"]:
                all_pass = False
                break

        if not all_pass:
            # Still record that this frontier point was checked and failed.
            checked.append(
                (
                    cfg,
                    {
                        "config": {
                            "dataset": cfg.dataset,
                            "dp_eps_train": cfg.dp_eps_train,
                            "dp_eps_cal": cfg.dp_eps_cal,
                            "nominal_coverage": cfg.nominal_coverage,
                            "coverage_target": cfg.coverage_target,
                        },
                        "formal": formal_result["formal"],
                        "empirical": {
                            "overall_empirical_ok": False,
                            "coverage_empirical_ok": False,
                        },
                    },
                )
            )
            continue

        summary = summarize_seed_results(cfg, seed_results, formal_result)
        checked.append((cfg, summary))
        feasible.append((cfg, summary))

    return checked, feasible, n


def choose_best(empirical_feasible: list[tuple[ExperimentConfig, dict]]):
    if not empirical_feasible:
        return None

    best = None
    best_key = None

    for cfg, result in empirical_feasible:
        key = objective(cfg, result)
        if best is None or key < best_key:
            best = (cfg, result)
            best_key = key

    return best


def compute_feasible_regions(contract: Contract, grid: SearchGrid) -> RegionResult:
    base = make_base_config()

    formal_checked, formal_feasible, n_formal = formal_search(base, contract, grid)
    best_failed_or_best = best_formal_candidate(formal_checked)

    if not formal_feasible:
        return RegionResult(
            formal_checked=formal_checked,
            formal_feasible=[],
            formal_frontier=[],
            empirical_checked=[],
            empirical_feasible=[],
            best_formal_candidate=best_failed_or_best,
            num_formal_evals=n_formal,
            num_empirical_evals=0,
            grid_size_total=len(formal_checked)
        )

    frontier = formal_frontier_candidates(formal_feasible)
    checked, empirical_feasible, n_emp = empirical_refine(frontier, grid.seeds)

    return RegionResult(
        formal_checked=formal_checked,
        formal_feasible=formal_feasible,
        formal_frontier=frontier,
        empirical_checked=checked,
        empirical_feasible=empirical_feasible,
        best_formal_candidate=best_failed_or_best,
        num_formal_evals=n_formal,
        num_empirical_evals=n_emp,
        grid_size_total=len(formal_checked)
    )

def compile_contract(
    contract: Contract,
    grid: SearchGrid,
    regions: RegionResult | None = None,
) -> CompileResult:
    if regions is None:
        regions = compute_feasible_regions(contract, grid)

    if not regions.formal_feasible:
        return CompileResult(
            status="FORMALLY_INFEASIBLE",
            config=None,
            formal_result=None,
            empirical_result=None,
            num_formal_evals=regions.num_formal_evals,
            num_empirical_evals=regions.num_empirical_evals,
            certificate_result={"certificate": None},
            compiler_mode="certificate_aware_frontier",
            frontier_rank_rule="increasing_certificate_width_then_empirical_objective",
            grid_size_total=len(regions.formal_checked),
        )

    best = choose_best(regions.empirical_feasible)
    if best is None:
        return CompileResult(
            status="EMPIRICALLY_INFEASIBLE",
            config=None,
            formal_result=None,
            empirical_result=None,
            num_formal_evals=regions.num_formal_evals,
            num_empirical_evals=regions.num_empirical_evals,
            compiler_mode="certificate_aware_frontier",
            frontier_rank_rule="increasing_certificate_width_then_empirical_objective",
            grid_size_total=len(regions.formal_checked),
        )

    cfg, result = best
    return CompileResult(
        status="FEASIBLE",
        config=cfg,
        formal_result={"formal": result["formal"]},
        empirical_result={"empirical": result["empirical"]},
        certificate_result={"certificate": result["certificate"]},
        num_formal_evals=regions.num_formal_evals,
        num_empirical_evals=regions.num_empirical_evals,
        compiler_mode="certificate_aware_frontier",
        frontier_rank_rule="increasing_certificate_width_then_empirical_objective",
        grid_size_total=len(regions.formal_checked),
    )

def print_summary(result: CompileResult):
    print(f"status: {result.status}")
    print(f"formal evals: {result.num_formal_evals}")
    print(f"empirical evals: {result.num_empirical_evals}")

    if result.config is None:
        return

    cfg = result.config
    print(
        "chosen config:",
        {
            "dp_eps_train": cfg.dp_eps_train,
            "dp_eps_cal": cfg.dp_eps_cal,
            "nominal_coverage": cfg.nominal_coverage,
            "coverage_target": cfg.coverage_target,
        },
    )
    print(
        "formal:",
        {
            "coverage_lower_bound_formal": result.formal_result["formal"]["coverage_lower_bound_formal"],
            "overall_formal_ok": result.formal_result["formal"]["overall_formal_ok"],
        },
    )
    print(
        "empirical:",
        {
            "test_coverage_dpcal": result.empirical_result["empirical"]["test_coverage_dpcal"],
            "avg_set_size_dpcal": result.empirical_result["empirical"]["avg_set_size_dpcal"],
            "tau_dp_cal": result.empirical_result["empirical"]["tau_dp_cal"],
        },
    )


def print_region_summary(regions: RegionResult):
    print(f"formal feasible: {len(regions.formal_feasible)}")
    print(f"formal frontier: {len(regions.formal_frontier)}")
    print(f"empirical checked: {len(regions.empirical_checked)}")
    print(f"empirical feasible: {len(regions.empirical_feasible)}")
    print(f"formal evals: {regions.num_formal_evals}")
    print(f"empirical evals: {regions.num_empirical_evals}")


def main():

    out_dir = "contracts/"
    ensure_dir(out_dir)

    grid = SearchGrid(
        dp_eps_train=[4.0],
        dp_eps_cal=[2.0, 4.0, 8.0],
        nominal_coverage=[0.55, 0.65, 0.75],
        cal_size=[1000, 2000, 4000],
        seeds=[0],
    )

    for target in [0.9]:
        contract = Contract(
            coverage_target=target,
            beta=1e-3,
            max_dp_eps_cal=8.0,
            max_dp_eps_train=4.0,
        )

        print(f"\n=== target={target} ===")
        regions = compute_feasible_regions(contract, grid)
        print_region_summary(regions)
        result = compile_contract(contract, grid, regions=regions)
        print_summary(result)
        card = build_contract_card(contract, grid, result, regions)
        print_contract_card(card)
        save_contract_card(card, f"{out_dir}/contract_card_target_{target}.json")

if __name__ == "__main__":
    main()
