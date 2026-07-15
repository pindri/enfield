import argparse

from utils import ensure_dir, load_reports, save_dataframe


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="toy_out_fmnist_main_mechanism_target070")
    ap.add_argument("--analysis_out_dir", type=str, default="analysis/fmnist_main_mechanism_target070")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.analysis_out_dir)

    df = load_reports(args.out_dir)
    save_dataframe(df, f"{args.analysis_out_dir}/all_reports.csv")

    print(f"Saved data from {args.out_dir} to {args.analysis_out_dir}/all_reports.csv")


if __name__ == "__main__":
    main()
