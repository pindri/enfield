from dataclasses import dataclass, replace
from toy import ExperimentConfig, run_experiment

@dataclass
class SimpleContract:
    coverage_target: float
    beta: float = 1e-3

@dataclass
class SearchGrid:
    dp_train_eps: list[float]
    dp_eps_cal: list[float]
    nominal_coverage: list[float]
    seeds: list[int]

@dataclass
class CompileResult:
    status: str
    config: ExperimentConfig | None
    formal_result: dict | None
    empirical_result: dict | None
    num_formal_evals: int
    num_empirical_evals: int


def make_base_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="cifar10",
        epochs_np=10,
        epochs_dp=4,
        cal_size=2000,
        batch_size=128,
        temperature=2.0,
        label_smoothing=0.0,
        write_report=False,
        verbose=False,
    )


def candidate_configs(base: ExperimentConfig, contract: SimpleContract, grid: SearchGrid):
    for teps in grid.dp_train_eps:
        for ceps in grid.dp_eps_cal:
            for nomcov in grid.nominal_coverage:
                yield replace(
                    base,
                    dp_train_eps=teps,
                    dp_eps_cal=ceps,
                    nominal_coverage=nomcov,
                    coverage_target=contract.coverage_target,
                    beta=contract.beta,
                )


def formal_search(base: ExperimentConfig, contract: SimpleContract, grid: SearchGrid):
    feasible = []
    n = 0
    # Only one seed needed for formal check.
    formal_seed = grid.seeds[0]

    for cfg in candidate_configs(base, contract, grid):
        cfg = replace(cfg, seed=formal_seed)
        result = run_experiment(cfg)
        n += 1
        if result["formal"]["overall_formal_ok"]:
            feasible.append((cfg, result))

    return feasible, n


def refine_empirically(formal_feasible, seeds: list[int]):
    n = 0
    if not formal_feasible:
        return None, n

    # Least conservative first.
    formal_feasible = sorted(
        formal_feasible,
        key=lambda x: (x[0].nominal_coverage, -x[0].dp_eps_cal, -x[0].dp_train_eps)
    )

    for cfg, formal_result in formal_feasible:
        empirical_ok_all = True
        last_result = None
        for seed in seeds:
            cfg_seed = replace(cfg, seed=seed)
            result = run_experiment(cfg_seed)
            n += 1
            last_result = result
            if not result["empirical"]["overall_empirical_ok"]:
                empirical_ok_all = False
                break
        if empirical_ok_all:
            return (cfg, formal_result, last_result), n

    return None, n


def compile_contract(contract: SimpleContract, grid: SearchGrid) -> CompileResult:
    base = make_base_config()

    formal_feasible, n_formal = formal_search(base, contract, grid)

    if not formal_feasible:
        return CompileResult(
            status="FORMALLY_INFEASIBLE",
            config=None,
            formal_result=None,
            empirical_result=None,
            num_formal_evals=n_formal,
            num_empirical_evals=0,
        )

    empirical_best, n_emp = refine_empirically(formal_feasible, grid.seeds)

    if empirical_best is None:
        return CompileResult(
            status="EMPIRICALLY_INFEASIBLE",
            config=None,
            formal_result=None,
            empirical_result=None,
            num_formal_evals=n_formal,
            num_empirical_evals=n_emp,
        )

    cfg, formal_result, empirical_result = empirical_best
    return CompileResult(
        status="FEASIBLE",
        config=cfg,
        formal_result=formal_result,
        empirical_result=empirical_result,
        num_formal_evals=n_formal,
        num_empirical_evals=n_emp,
    )


def main():
    contract = SimpleContract(coverage_target=0.8, beta=1e-3)
    grid = SearchGrid(
        dp_train_eps=[2.0, 4.0, 8.0],
        dp_eps_cal=[1.0, 2.0, 4.0, 8.0],
        nominal_coverage=[0.7, 0.8, 0.9],
        seeds=[0, 1, 2],
    )

    result = compile_contract(contract, grid)
    print(result)


if __name__ == "__main__":
    main()
