from compiler import Contract, CompileResult, RegionResult, SearchGrid
import json


def build_contract_card(contract: Contract,
                        grid: SearchGrid,
                        result: CompileResult,
                        regions: RegionResult | None = None) -> dict:
    card = {
        "request": {
            "coverage_target": contract.coverage_target,
            "beta": contract.beta,
            "max_dp_eps_train": contract.max_dp_eps_train,
            "max_dp_eps_cal": contract.max_dp_eps_cal,
        },
        "decision": {
            "status": result.status,
        },
    }

    card["search_space"] = {
        "dp_train_eps": grid.dp_train_eps,
        "dp_eps_cal": grid.dp_eps_cal,
        "nominal_coverage": grid.nominal_coverage,
        "seed": grid.seeds,
    }

    if result.config is not None:
        cfg = result.config
        card["decision"]["chosen_config"] = {
            "dp_train_eps": cfg.dp_train_eps,
            "dp_eps_cal": cfg.dp_eps_cal,
            "nominal_coverage": cfg.nominal_coverage,
            "epochs_dp": cfg.epochs_dp,
            "cal_size": cfg.cal_size,
            "temperature": cfg.temperature,
            "label_smoothing": cfg.label_smoothing,
        }
        card["formal_certificate"] = result.formal_result["formal"]
        card["empirical_validation"] = result.empirical_result["empirical"]

    if regions is not None:
        card["search_summary"] = {
            "formal_feasible_count": len(regions.formal_feasible),
            "formal_frontier_count": len(regions.formal_frontier),
            "empirical_checked_count": len(regions.empirical_checked),
            "empirical_feasible_count": len(regions.empirical_feasible),
            "formal_evals": regions.num_formal_evals,
            "empirical_evals": regions.num_empirical_evals,
        }

    return card

def print_contract_card(card: dict):

    def fmt_value(v):
        if isinstance(v, list):
            return "{" + ", ".join(str(x) for x in v) + "}"
        return str(v)


    print("\nModel Contract Card")
    print("===================")

    print("\nRequest")
    print("-------")
    for k, v in card["request"].items():
        print(f"{k}: {fmt_value(v)}")

    print("\nCompiler decision")
    print("-----------------")
    print(f"status: {card['decision']['status']}")

    if "chosen_config" in card["decision"]:
        print("chosen_config:")
        for k, v in card["decision"]["chosen_config"].items():
            print(f"{k}: {fmt_value(v)}")

    if "formal_certificate" in card:
        print("\nFormal certificate")
        print("------------------")
        for k, v in card["formal_certificate"].items():
            print(f"{k}: {fmt_value(v)}")

    if "empirical_validation" in card:
        print("\nEmpirical validation")
        print("--------------------")
        for k, v in card["empirical_validation"].items():
            print(f"{k}: {fmt_value(v)}")

    if "search_summary" in card:
        print("\nSearch summary")
        print("--------------")
        for k, v in card["search_summary"].items():
            print(f"{k}: {fmt_value(v)}")

    if "search_space" in card:
        print("\nSearch space")
        print("------------")
        for k, v in card["search_space"].items():
            print(f"{k}: {fmt_value(v)}")


def save_contract_card(card: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, sort_keys=True)