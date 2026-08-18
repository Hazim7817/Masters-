from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import ttest_rel


# ======================================================
# SETTINGS
# ======================================================

RECORDING_DURATION_MIN = 10
# Use 10 if analysing 10 min post-stimulation only
# Use 20 if analysing the full 20 min recording

DAY_FOLDERS = [
    (Path(r"D:\Tarmizi\Analysis\1 July"), "1 July"),
    (Path(r"D:\Tarmizi\Analysis\7 July"), "7 July"),
    (Path(r"D:\Tarmizi\Analysis\22 July"), "22 July"),
]

OUTPUT_DIR = Path(r"D:\Tarmizi\Analysis\Paired_C5a_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# LOAD INDIVIDUAL SUMMARY FILES
# ======================================================

def extract_group_and_repeat(file_name):
    file_name = file_name.replace(".csv", "")

    control_match = re.search(r"Control_(\d+)_summary", file_name)
    c5a_match = re.search(r"With_C5a_(\d+)_summary", file_name)

    if control_match:
        return "Control", control_match.group(1)

    if c5a_match:
        return "C5a stimulated", c5a_match.group(1)

    return None, None


def load_individual_summary_files(day_folders):
    all_rows = []

    for folder, day_label in day_folders:

        summary_files = list(folder.glob("*_summary.csv"))

        for file in summary_files:

            # Skip already-combined summary files
            if file.name.startswith("combined_condition_summary"):
                continue

            if file.name.startswith("condition_level_summary"):
                continue

            group, repeat = extract_group_and_repeat(file.name)

            if group is None:
                continue

            temp = pd.read_csv(file)

            temp["day"] = day_label
            temp["group"] = group
            temp["repeat"] = repeat
            temp["file_name"] = file.name
            temp["pair_id"] = day_label + "_repeat_" + repeat

            if "condition" not in temp.columns:
                temp["condition"] = group

            all_rows.append(temp)

    if len(all_rows) == 0:
        raise ValueError(
            "No individual summary files found. "
            "Check that files are named like Control_1_summary.csv and With_C5a_1_summary.csv."
        )

    summary = pd.concat(all_rows, ignore_index=True)

    if "percent_pulsing" not in summary.columns:
        summary["percent_pulsing"] = (
            100 * summary["n_pulsing"] / summary["n_total"]
        )

    if "n_non_pulsing" not in summary.columns:
        summary["n_non_pulsing"] = (
            summary["n_total"] - summary["n_pulsing"]
        )

    return summary


# ======================================================
# ADD MEAN PULSE FREQUENCY ONLY AMONG PULSING CELLS
# ======================================================

def add_mean_pulse_frequency_pulsing_cells(summary):
    summary = summary.copy()

    # This calculates pulse frequency ONLY among pulsing cells.
    #
    # Assumption:
    # mean_peak_count = mean number of detected peaks per cell across ALL cells.
    #
    # Therefore:
    # total peaks = mean_peak_count * n_total
    # mean peaks per pulsing cell = total peaks / n_pulsing
    # mean pulse frequency among pulsing cells =
    # mean peaks per pulsing cell / recording duration

    if "mean_peak_count" in summary.columns:

        mean_peak_count = pd.to_numeric(
            summary["mean_peak_count"],
            errors="coerce"
        )

        n_total = pd.to_numeric(
            summary["n_total"],
            errors="coerce"
        )

        n_pulsing = pd.to_numeric(
            summary["n_pulsing"],
            errors="coerce"
        )

        total_peaks = mean_peak_count * n_total

        summary["mean_pulse_frequency_pulsing_cells_per_min"] = np.where(
            n_pulsing > 0,
            (total_peaks / n_pulsing) / RECORDING_DURATION_MIN,
            np.nan
        )

        print(
            "Calculated mean_pulse_frequency_pulsing_cells_per_min "
            "using mean_peak_count, n_total, n_pulsing, and recording duration."
        )

    else:
        summary["mean_pulse_frequency_pulsing_cells_per_min"] = np.nan

        print(
            "WARNING: mean_peak_count column not found. "
            "Mean pulse frequency among pulsing cells could not be calculated."
        )

    return summary


# ======================================================
# POOLED CHI-SQUARE / FISHER TEST
# ======================================================

def compare_pulsing_counts(summary):
    control = summary[summary["group"] == "Control"]
    c5a = summary[summary["group"] == "C5a stimulated"]

    control_pulsing = int(control["n_pulsing"].sum())
    control_non = int(control["n_non_pulsing"].sum())

    c5a_pulsing = int(c5a["n_pulsing"].sum())
    c5a_non = int(c5a["n_non_pulsing"].sum())

    table = [
        [control_pulsing, control_non],
        [c5a_pulsing, c5a_non]
    ]

    chi2, p, dof, expected = chi2_contingency(table)
    fisher_odds_ratio, fisher_p = fisher_exact(table)

    n = np.sum(table)
    cramers_v = np.sqrt(chi2 / n)

    result = pd.DataFrame([{
        "comparison": "Control vs C5a stimulated",
        "control_pulsing": control_pulsing,
        "control_non_pulsing": control_non,
        "c5a_pulsing": c5a_pulsing,
        "c5a_non_pulsing": c5a_non,
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "chi_square_p_value": p,
        "cramers_v": cramers_v,
        "fisher_odds_ratio": fisher_odds_ratio,
        "fisher_p_value": fisher_p
    }])

    return result, table


def plot_stacked_counts(table, outfile):
    labels = ["Control", "C5a stimulated"]

    pulsing = [table[0][0], table[1][0]]
    non = [table[0][1], table[1][1]]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(labels, non, label="Non-pulsing")
    ax.bar(labels, pulsing, bottom=non, label="Pulsing")

    for i in range(2):
        ax.text(
            i,
            non[i] / 2,
            str(non[i]),
            ha="center",
            va="center",
            fontsize=10
        )

        ax.text(
            i,
            non[i] + pulsing[i] / 2,
            str(pulsing[i]),
            ha="center",
            va="center",
            fontsize=10
        )

    ax.set_ylabel("Number of cells", fontsize=13)
    ax.set_title("Pulsing vs non-pulsing cells", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()

    print(f"Saved: {outfile}")


# ======================================================
# PAIRED T-TEST
# ======================================================

def make_paired_table(summary, metric):
    paired = summary.pivot_table(
        index=["day", "repeat", "pair_id"],
        columns="group",
        values=metric,
        aggfunc="first"
    ).reset_index()

    paired = paired.dropna(subset=["Control", "C5a stimulated"])

    return paired


def run_paired_ttest(summary, metric):
    paired = make_paired_table(summary, metric)

    if len(paired) < 2:
        raise ValueError(f"Not enough paired samples for {metric}")

    t_stat, p_value = ttest_rel(
        paired["C5a stimulated"],
        paired["Control"],
        nan_policy="omit"
    )

    result = pd.DataFrame([{
        "metric": metric,
        "test": "Paired t-test",
        "n_pairs": len(paired),
        "control_mean": paired["Control"].mean(),
        "control_sem": paired["Control"].sem(),
        "stimulated_mean": paired["C5a stimulated"].mean(),
        "stimulated_sem": paired["C5a stimulated"].sem(),
        "mean_difference_c5a_minus_control": (
            paired["C5a stimulated"] - paired["Control"]
        ).mean(),
        "t_statistic": t_stat,
        "p_value": p_value
    }])

    return paired, result


# ======================================================
# PLOT HELPERS
# ======================================================

def add_mean_sem(ax, x, values, width=0.2):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    mean = np.mean(values)

    if len(values) > 1:
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
    else:
        sem = 0

    ax.hlines(mean, x - width, x + width, linewidth=2)
    ax.errorbar(x, mean, yerr=sem, fmt="none", capsize=5)


# ======================================================
# PLOT 1: PERCENT PULSING DOT PLOT COLOURED BY DAY
# ======================================================

def plot_percent_pulsing_dotplot(summary, output_dir):
    group_order = ["Control", "C5a stimulated"]
    day_order = ["1 July", "7 July", "22 July"]
    day_order = [day for day in day_order if day in summary["day"].unique()]

    x_positions = {
        "Control": 0,
        "C5a stimulated": 1
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    rng = np.random.default_rng(1)

    cmap = plt.get_cmap("tab10")
    day_colours = {
        day: cmap(i)
        for i, day in enumerate(day_order)
    }

    for day in day_order:
        for group in group_order:

            subset = summary[
                (summary["day"] == day)
                & (summary["group"] == group)
            ]

            if subset.empty:
                continue

            xpos = x_positions[group]
            y = subset["percent_pulsing"].to_numpy(dtype=float)

            jitter = rng.normal(0, 0.06, size=len(y))

            ax.scatter(
                np.full(len(y), xpos) + jitter,
                y,
                s=70,
                color=day_colours[day],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                label=day if group == "Control" else None,
                zorder=3
            )

    for group in group_order:
        y = summary.loc[
            summary["group"] == group,
            "percent_pulsing"
        ].to_numpy(dtype=float)

        add_mean_sem(
            ax=ax,
            x=x_positions[group],
            values=y,
            width=0.18
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_order, fontsize=12)

    ax.set_ylabel("Percent pulsing (%)", fontsize=13)
    ax.set_xlabel("")
    ax.set_title("Percent pulsing in Control vs C5a-stimulated cells", fontsize=14)

    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        title="Day",
        loc="upper right"
    )

    plt.tight_layout()

    outfile = output_dir / "percent_pulsing_C5a_vs_control_dotplot_by_day.png"
    plt.savefig(outfile, dpi=300)
    plt.show()

    print(f"Saved: {outfile}")


# ======================================================
# PLOT 2: PERCENT PULSING BAR PLOT WITH PAIRED LINES
# ======================================================

def plot_percent_pulsing_paired_barplot(paired_df, output_dir):
    group_order = ["Control", "C5a stimulated"]

    means = [paired_df[group].mean() for group in group_order]
    sems = [paired_df[group].sem() for group in group_order]

    x = np.arange(len(group_order))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        alpha=0.8
    )

    for _, row in paired_df.iterrows():
        y_control = row["Control"]
        y_c5a = row["C5a stimulated"]

        ax.plot(
            [0, 1],
            [y_control, y_c5a],
            color="black",
            alpha=0.4,
            linewidth=1
        )

        ax.scatter(
            [0, 1],
            [y_control, y_c5a],
            color="black",
            s=50,
            zorder=3
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=12)

    ax.set_ylabel("Percent pulsing (%)", fontsize=13)
    ax.set_title("Percent pulsing in Control vs C5a-stimulated cells", fontsize=14)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    outfile = output_dir / "percent_pulsing_paired_barplot.png"
    plt.savefig(outfile, dpi=300)
    plt.show()

    print(f"Saved: {outfile}")


# ======================================================
# PLOT 3: MEAN PULSE FREQUENCY IN PULSING CELLS BAR PLOT
# ======================================================

def plot_mean_pulse_frequency_pulsing_cells_barplot(paired_df, output_dir):
    group_order = ["Control", "C5a stimulated"]

    means = [paired_df[group].mean() for group in group_order]
    sems = [paired_df[group].sem() for group in group_order]

    x = np.arange(len(group_order))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=12)

    ax.set_ylabel(
        "Mean pulse frequency in pulsing cells\n(pulses/frame)",
        fontsize=13
    )

    ax.set_title(
        "Mean pulse frequency in pulsing cells",
        fontsize=14
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    outfile = output_dir / "mean_pulse_frequency_pulsing_cells_barplot.png"
    plt.savefig(outfile, dpi=300)
    plt.show()

    print(f"Saved: {outfile}")


# ======================================================
# MAIN SCRIPT
# ======================================================

if __name__ == "__main__":

    summary = load_individual_summary_files(DAY_FOLDERS)
    summary = add_mean_pulse_frequency_pulsing_cells(summary)

    print("\nLoaded individual summary files:")
    print(
        summary[
            [
                "day",
                "group",
                "repeat",
                "pair_id",
                "file_name",
                "n_total",
                "n_pulsing",
                "n_non_pulsing",
                "percent_pulsing",
                "mean_peak_count",
                "mean_pulse_frequency_pulsing_cells_per_min"
            ]
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "all_individual_summary_files_with_pair_ids.csv",
        index=False
    )

    # ---------------------------
    # Pooled pulsing count stats
    # ---------------------------

    count_results, count_table = compare_pulsing_counts(summary)

    count_results.to_csv(
        OUTPUT_DIR / "Chi_square_and_Fisher_pulsing_counts.csv",
        index=False
    )

    plot_stacked_counts(
        count_table,
        OUTPUT_DIR / "Pulsing_vs_nonpulsing_stacked_counts.png"
    )

    print("\nChi-square / Fisher results:")
    print(count_results)

    # ---------------------------
    # Paired t-tests
    # ---------------------------

    paired_percent, result_percent = run_paired_ttest(
        summary,
        metric="percent_pulsing"
    )

    paired_frequency, result_frequency = run_paired_ttest(
        summary,
        metric="mean_pulse_frequency_pulsing_cells_per_min"
    )

    paired_percent.to_csv(
        OUTPUT_DIR / "paired_percent_pulsing_table.csv",
        index=False
    )

    paired_frequency.to_csv(
        OUTPUT_DIR / "paired_mean_pulse_frequency_pulsing_cells_table.csv",
        index=False
    )

    result_percent.to_csv(
        OUTPUT_DIR / "percent_pulsing_paired_ttest.csv",
        index=False
    )

    result_frequency.to_csv(
        OUTPUT_DIR / "mean_pulse_frequency_pulsing_cells_paired_ttest.csv",
        index=False
    )

    paired_results = pd.concat(
        [result_percent, result_frequency],
        ignore_index=True
    )

    paired_results.to_csv(
        OUTPUT_DIR / "paired_ttest_summary_table.csv",
        index=False
    )

    print("\nPaired t-test summary:")
    print(paired_results)

    # ---------------------------
    # Plots
    # ---------------------------

    plot_percent_pulsing_dotplot(
        summary,
        OUTPUT_DIR
    )

    plot_percent_pulsing_paired_barplot(
        paired_percent,
        OUTPUT_DIR
    )

    plot_mean_pulse_frequency_pulsing_cells_barplot(
        paired_frequency,
        OUTPUT_DIR
    )

    print("\nFinished.")
    print(f"Saved all outputs to: {OUTPUT_DIR}")

    
"""
    comparisons = [

        (
            "Without_LPS_Without_LY_Without_C5a",
            "Without_LPS_Without_LY_With_C5a"
        ),

        (
            "Without_LPS_Without_LY_Without_C5a",
            "With_LPS_Without_LY_Without_C5a"
        ),

        (
            "Without_LPS_Without_LY_Without_C5a",
            "Without_LPS_With_LY_Without_C5a"
        ),

        (
            "With_LPS_Without_LY_Without_C5a",
            "With_LPS_Without_LY_With_C5a"
        ),

        (
            "Without_LPS_Without_LY_With_C5a",
            "With_LPS_Without_LY_With_C5a"
        )
"""