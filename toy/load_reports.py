import argparse

from utils import ensure_dir, load_reports, save_dataframe


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="toy_out_saturation_check_target070")
    ap.add_argument("--analysis_out_dir", type=str, default="analysis/saturation_check_target07")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.analysis_out_dir)

    # Check and create folders.
    report_dir = args.out_dir
    analysis_out_dir = args.analysis_out_dir

    df = load_reports(report_dir)
    save_dataframe(df, f"{analysis_out_dir}/all_reports.csv")

    print(f"Saved data from {report_dir} to {analysis_out_dir}/all_reports.csv")



if __name__ == "__main__":
    main()
