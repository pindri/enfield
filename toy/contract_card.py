from compiler import Contract, CompileResult, RegionResult, SearchGrid
import json

def build_contract_card(contract: Contract,
                        grid: SearchGrid,
                        result: CompileResult,
                        regions: RegionResult | None = None) -> dict:
    card = {
        "00_request": {
            "coverage_target": contract.coverage_target,
            "beta": contract.beta,
            "max_dp_eps_train": contract.max_dp_eps_train,
            "max_dp_eps_cal": contract.max_dp_eps_cal,
            "objective" : contract.objective,
        },
        "04_search_space": {
            "01_grid": {
                "dp_eps_train": grid.dp_eps_train,
                "dp_eps_cal": grid.dp_eps_cal,
                "nominal_coverage": grid.nominal_coverage,
                "seeds": grid.seeds,
            },
        },
        "01_decision": {
            "00_status": result.status,
        },
        "02_evidence": {},
    }

    if regions is not None:
        card["04_search_space"]["00_search_summary"] = {
            "formal_feasible_count": len(regions.formal_feasible),
            "formal_frontier_count": len(regions.formal_frontier),
            "empirical_checked_count": len(regions.empirical_checked),
            "empirical_feasible_count": len(regions.empirical_feasible),
            "num_formal_evals": regions.num_formal_evals,
            "num_empirical_evals": regions.num_empirical_evals,
        }

    # FEASIBLE: chosen witness + full evidence.
    if result.status == "FEASIBLE" and result.config is not None:
        cfg = result.config
        card["01_decision"]["01_chosen_config"] = {
            "dp_eps_train": cfg.dp_eps_train,
            "dp_eps_cal": cfg.dp_eps_cal,
            "nominal_coverage": cfg.nominal_coverage,
            "coverage_target": cfg.coverage_target,
            "cal_size": cfg.cal_size,
            "epochs_dp": cfg.epochs_dp,
            "temperature": cfg.temperature,
            "label_smoothing": cfg.label_smoothing,
        }
        card["02_evidence"] = {
            "formal_guarantees": result.formal_result["formal"],
            "empirical_validation": result.empirical_result["empirical"],
            "margins": {
                "coverage_formal_margin": round(
                    result.formal_result["formal"]["coverage_lower_bound_formal"] - contract.coverage_target, 5
                ),
                "coverage_empirical_margin": round(
                    result.empirical_result["empirical"]["test_coverage_dpcal"] - contract.coverage_target, 5
                ),
                "training_privacy_formal_margin": round(contract.max_dp_eps_train - cfg.dp_eps_train, 5),
                "calibration_privacy_formal_margin": round(contract.max_dp_eps_cal - cfg.dp_eps_cal, 5),
            },
        }
        return card

    # FORMALLY_INFEASIBLE: show best formal attempt and why it failed.
    if result.status == "FORMALLY_INFEASIBLE":
        if regions is not None and regions.best_formal_candidate is not None:
            cfg, fres = regions.best_formal_candidate
            lb = fres["formal"]["coverage_lower_bound_formal"]
            card["02_evidence"]["00_best_formal_attempt"] = {
                "config": {
                    "dp_eps_train": cfg.dp_eps_train,
                    "dp_eps_cal": cfg.dp_eps_cal,
                    "nominal_coverage": cfg.nominal_coverage,
                    "coverage_target": cfg.coverage_target,
                },
                "formal_guarantees": fres["formal"],
                "margins" : {
                    # "coverage_requested_target": contract.coverage_target,
                    "coverage_formal_margin": round(lb - contract.coverage_target, 5),
                    "training_privacy_formal_margin": round(contract.max_dp_eps_train - cfg.dp_eps_train, 5),
                    "calibration_privacy_formal_margin": round(contract.max_dp_eps_cal - cfg.dp_eps_cal, 5),
                }
            }
        else:
            card["02_evidence"]["00_best_formal_attempt"] = None
        return card

    # EMPIRICALLY_INFEASIBLE: formal candidates existed, but checked frontier failed empirically.
    if result.status == "EMPIRICALLY_INFEASIBLE":
        if regions is not None:
            frontier_summary = []
            for cfg, r in regions.empirical_checked:
                frontier_summary.append(
                    {
                        "config": {
                            "dp_eps_train": cfg.dp_eps_train,
                            "dp_eps_cal": cfg.dp_eps_cal,
                            "nominal_coverage": cfg.nominal_coverage,
                            "coverage_target": cfg.coverage_target,
                        },
                        "formal_guarantees": r.get("formal", {}),
                        "empirical_validation": r.get("empirical", {}),
                    }
                )
            card["02_evidence"]["01_checked_frontier"] = frontier_summary

            if regions.best_formal_candidate is not None:
                cfg, fres = regions.best_formal_candidate
                card["02_evidence"]["00_best_formal_attempt"] = {
                    "config": {
                        "dp_eps_train": cfg.dp_eps_train,
                        "dp_eps_cal": cfg.dp_eps_cal,
                        "nominal_coverage": cfg.nominal_coverage,
                        "coverage_target": cfg.coverage_target,
                    },
                    "formal_guarantees": fres["formal"],
                }
        return card

    return card


def print_contract_card(card: dict):
    def fmt_value(v):
        if isinstance(v, list):
            return "{" + ", ".join(str(x) for x in v) + "}"
        if isinstance(v, float):
            return f"{v:.5f}"
        return str(v)

    def print_block(title: str, d: dict, indent: int = 0):
        prefix = " " * indent
        print(f"\n{prefix}{title}")
        print(f"{prefix}{'-' * len(title)}")
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                print_block("", v, indent + 2)
            else:
                print(f"{prefix}{k}: {fmt_value(v)}")

    print("\nModel Contract Card")
    print("===================")

    if "00_request" in card:
        print_block("Request", card["00_request"])

    if "01_decision" in card:
        print("\nDecision")
        print("--------")
        print(f"00_status: {card['01_decision']['00_status']}")
        if "01_chosen_config" in card["01_decision"]:
            print("\nChosen configuration")
            print("--------------------")
            for k, v in card["01_decision"]["01_chosen_config"].items():
                print(f"{k}: {fmt_value(v)}")

    if "02_evidence" in card and card["02_evidence"]:
        ev = card["02_evidence"]
        if "formal_guarantees" in ev:
            print("\nFormal guarantees")
            print("-----------------")
            for k, v in ev["formal_guarantees"].items():
                print(f"{k}: {fmt_value(v)}")
        if "empirical_validation" in ev:
            print("\nEmpirical validation")
            print("--------------------")
            for k, v in ev["empirical_validation"].items():
                print(f"{k}: {fmt_value(v)}")
        if "margins" in ev:
            print("\nMargins")
            print("-------")
            for k, v in ev["margins"].items():
                print(f"{k}: {fmt_value(v)}")

        if "00_best_formal_attempt" in ev and ev["00_best_formal_attempt"] is not None:
            bfa = ev["00_best_formal_attempt"]
            print("\nBest formal attempt")
            print("-------------------")
            if "config" in bfa:
                print("config:")
                for k, v in bfa["config"].items():
                    print(f"  {k}: {fmt_value(v)}")
            if "formal_guarantees" in bfa:
                print("formal_guarantees:")
                for k, v in bfa["formal_guarantees"].items():
                    print(f"  {k}: {fmt_value(v)}")
            if "margins" in bfa:
                print("margins:")
                for k, v in bfa["margins"].items():
                    print(f"  {k}: {fmt_value(v)}")

        if "01_checked_frontier" in ev:
            frontier = ev["01_checked_frontier"]
            print("\nChecked frontier")
            print("----------------")
            print(f"num_checked_frontier_points: {len(frontier)}")
            for i, point in enumerate(frontier, start=1):
                print(f"\nfrontier_point_{i}:")
                if "config" in point:
                    print("  config:")
                    for k, v in point["config"].items():
                        print(f"    {k}: {fmt_value(v)}")
                if "formal_guarantees" in point:
                    print("  formal_guarantees:")
                    for k, v in point["formal_guarantees"].items():
                        print(f"    {k}: {fmt_value(v)}")
                if "empirical_validation" in point:
                    print("  empirical_validation:")
                    for k, v in point["empirical_validation"].items():
                        print(f"    {k}: {fmt_value(v)}")

    if "04_search_space" in card:
        ss = card["04_search_space"]
        if "00_search_summary" in ss:
            print("\nSearch summary")
            print("--------------")
            for k, v in ss["00_search_summary"].items():
                print(f"{k}: {fmt_value(v)}")
        if "01_grid" in ss:
            print("\nSearch grid")
            print("-----------")
            for k, v in ss["01_grid"].items():
                print(f"{k}: {fmt_value(v)}")

    # SOme overall printing.
    if card["01_decision"]["00_status"] == "FEASIBLE":
        print("\nInterpretation")
        print("--------------")
        print("A feasible configuration was found within the searched grid.")
    elif card["01_decision"]["00_status"] == "FORMALLY_INFEASIBLE":
        print("\nInterpretation")
        print("--------------")
        print("No configuration in the searched grid satisfies the formal contract.")
    elif card["01_decision"]["00_status"] == "EMPIRICALLY_INFEASIBLE":
        print("\nInterpretation")
        print("--------------")
        print("Some configurations were formally admissible, but none of the checked frontier points satisfied the empirical contract.")



def save_contract_card(card: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, sort_keys=True)