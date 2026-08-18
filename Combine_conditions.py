from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# ASSIGN EACH FILE TO AN EXPERIMENTAL CONDITION
# ============================================================

def assign_condition(file_name: str) -> str:
    """
    Determine the experimental condition from the filename.

    The filename is searched for indicators describing whether
    LPS, LY and C5a were present during the experiment.

    Example filenames
    -----------------
    With_LPS_With_C5a_1_summary.csv
    Without_LPS_With_LY_Without_C5a_summary.csv

    Parameters
    ----------
    file_name : str
        Name of the summary CSV file.

    Returns
    -------
    str
        Standardised condition name describing the presence or
        absence of LPS, LY and C5a.
    """

    # Convert filename to lowercase so matching is
    # independent of capitalisation
    name = file_name.lower()

    # Determine whether each treatment is present
    has_lps = "with_lps" in name
    has_ly = "with_ly" in name
    has_c5a = "with_c5a" in name

    # Convert Boolean treatment status into readable labels
    lps_status = "With_LPS" if has_lps else "Without_LPS"
    ly_status = "With_LY" if has_ly else "Without_LY"
    c5a_status = "With_C5a" if has_c5a else "Without_C5a"

    # Combine treatment information into one condition label
    return f"{lps_status}_{ly_status}_{c5a_status}"


# ============================================================
# COMBINE SUMMARY FILES FROM MULTIPLE EXPERIMENTS
# ============================================================

def combine_summary_files(input_dir: Path, output_dir: Path):
    """
    Combine per-experiment calcium-imaging summary files and
    generate condition-level statistics.

    This function reads all files ending in '_summary.csv' from
    the specified input directory. Each file represents one
    experimental repeat that has already been analysed using the
    individual-cell calcium-analysis script.

    The script:
        1. identifies all experiment summary files;
        2. assigns each experiment to a treatment condition;
        3. combines all experiments into a master dataframe;
        4. recalculates the proportion of pulsing cells;
        5. groups experimental repeats by condition;
        6. calculates mean, median, SD and SEM across repeats;
        7. calculates the pooled percentage of pulsing cells.

    Parameters
    ----------
    input_dir : Path
        Directory containing the per-experiment '*_summary.csv'
        files.

    output_dir : Path
        Directory in which the combined output files will be saved.

    Returns
    -------
    master_df : pd.DataFrame
        Experiment-level dataframe containing one row for each
        experimental repeat.

    condition_summary : pd.DataFrame
        Condition-level dataframe summarising all experimental
        repeats belonging to the same treatment condition.
    """

    # Create the output directory if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all summary CSV files in the input directory
    summary_files = sorted(
        input_dir.glob("*_summary.csv")
    )

    # Stop the analysis if no summary files are present
    if len(summary_files) == 0:
        raise FileNotFoundError(
            "No summary CSV files found."
        )

    # Store dataframes from each experimental repeat
    all_data = []

    # ========================================================
    # LOAD EACH EXPERIMENTAL SUMMARY
    # ========================================================

    for file in summary_files:

        # Read one experiment-level summary file
        df = pd.read_csv(file)

        # Remove "_summary" from the filename to create
        # an experiment identifier
        experiment_name = file.stem.replace(
            "_summary",
            ""
        )

        # Record the experiment identifier
        df["experiment"] = experiment_name

        # Record the source summary file
        df["summary_file"] = file.name

        # Assign the experiment to its treatment condition
        df["condition"] = assign_condition(file.name)

        all_data.append(df)

    # Combine all experimental repeats into one dataframe
    master_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # ========================================================
    # RECALCULATE PULSING CELL STATISTICS
    # ========================================================

    # Number of cells without a detected calcium peak
    master_df["n_non_pulsing"] = (
        master_df["n_total"]
        - master_df["n_pulsing"]
    )

    # Percentage of cells classified as pulsatile within
    # each experimental repeat
    master_df["percent_pulsing"] = (
        100
        * master_df["n_pulsing"]
        / master_df["n_total"]
    )

    # Sort experiments first by condition and then by
    # experiment identifier
    master_df = master_df.sort_values(
        ["condition", "experiment"]
    )

    # Save the combined experiment-level dataframe
    master_df.to_csv(
        output_dir / "combined_condition_summary.csv",
        index=False
    )

    # ========================================================
    # CONDITION-LEVEL SUMMARY
    # ========================================================

    condition_summary = (
        master_df
        .groupby("condition", as_index=False)
        .agg(

            # Number of independent experimental repeats
            n_experiments=(
                "experiment",
                "count"
            ),

            # Total number of analysed cells across repeats
            total_cells=(
                "n_total",
                "sum"
            ),

            # Total number of cells classified as pulsatile
            total_pulsing=(
                "n_pulsing",
                "sum"
            ),

            # Total number of non-pulsing cells
            total_non_pulsing=(
                "n_non_pulsing",
                "sum"
            ),

            # Mean percentage pulsing across experimental repeats.
            #
            # Importantly, each experiment contributes equally
            # regardless of how many cells were analysed.
            mean_percent_pulsing=(
                "percent_pulsing",
                "mean"
            ),

            # Median percentage pulsing across repeats
            median_percent_pulsing=(
                "percent_pulsing",
                "median"
            ),

            # Standard deviation between experimental repeats
            std_percent_pulsing=(
                "percent_pulsing",
                "std"
            ),

            # Standard error of the mean across independent
            # experimental repeats
            sem_percent_pulsing=(
                "percent_pulsing",
                lambda x:
                    x.std(ddof=1) / np.sqrt(len(x))
            ),

            # Average number of calcium peaks per cell
            # across experimental summaries
            mean_peak_count=(
                "mean_peak_count",
                "mean"
            ),

            # Mean coefficient of variation across repeats
            mean_cv=(
                "mean_cv",
                "mean"
            ),

            # Mean fluorescence area under the curve
            mean_auc=(
                "mean_auc",
                "mean"
            ),

            # Mean maximum fluorescence intensity
            mean_max_intensity=(
                "mean_max_intensity",
                "mean"
            )
        )
    )

    # ========================================================
    # POOLED PERCENTAGE OF PULSING CELLS
    # ========================================================

    # Calculate the percentage pulsing after pooling all cells
    # belonging to the same condition.
    #
    # Unlike mean_percent_pulsing, this gives greater weight to
    # experiments containing larger numbers of analysed cells.
    condition_summary["pooled_percent_pulsing"] = (
        100
        * condition_summary["total_pulsing"]
        / condition_summary["total_cells"]
    )

    # Save the condition-level summary table
    condition_summary.to_csv(
        output_dir / "condition_level_summary.csv",
        index=False
    )

    # ========================================================
    # PRINT RESULTS TO CONSOLE
    # ========================================================

    print("\n==============================")
    print("Combined Experiments")
    print("==============================")
    print(master_df)

    print("\n==============================")
    print("Condition Summary")
    print("==============================")
    print(condition_summary)

    print("\nSaved files:")

    print(
        output_dir /
        "combined_condition_summary.csv"
    )

    print(
        output_dir /
        "condition_level_summary.csv"
    )

    return master_df, condition_summary


# ============================================================
# RUN ANALYSIS
# ============================================================

if __name__ == "__main__":

    # Directory containing the individual experiment
    # summary files generated by the previous analysis script
    input_dir = Path(
        r"D:\Tarmizi\Analysis\7 July"
    )

    # Directory in which the combined summaries will be saved
    output_dir = Path(
        r"D:\Tarmizi\Analysis\7 July"
    )

    # Combine all experimental repeats and calculate
    # condition-level summary statistics
    master_df, condition_summary = combine_summary_files(
        input_dir=input_dir,
        output_dir=output_dir
    )