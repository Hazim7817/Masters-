from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# OPTIONAL ΔF/F0 NORMALISATION
# ============================================================
# Only use this if you dont normalise the data before peak detection.
def normalise_dff(df: pd.DataFrame, baseline_frames: int = 3) -> pd.DataFrame:
    """
    Convert raw fluorescence intensities to ΔF/F0.

    F0 is calculated independently for each cell as the mean
    fluorescence intensity across the first 'baseline_frames'
    frames.

    ΔF/F0 = (F - F0) / F0

    Parameters
    ----------
    df : pd.DataFrame
        Fluorescence intensity data.
        Rows represent image frames and columns represent cells.

    baseline_frames : int
        Number of initial frames used to calculate F0.

    Returns
    -------
    pd.DataFrame
        ΔF/F0-normalised fluorescence traces.

    Notes
    -----
    This function is currently defined for optional use but is
    not called in the main analysis below. Therefore, the current
    peak-detection analysis is performed on raw fluorescence
    intensity values.
    """

    # Calculate baseline fluorescence independently for each cell
    f0 = df.iloc[:baseline_frames].mean(axis=0)

    # Calculate ΔF/F0 for every frame
    dff = (df - f0) / f0

    return dff


# ============================================================
# LOAD CALCIUM-IMAGING DATA
# ============================================================

def load_calcium_csv(path: Path) -> pd.DataFrame:
    """
    Load fluorescence-intensity measurements from a CSV file.

    The expected format is:
        - rows = successive imaging frames
        - columns = individual cells

    Non-numeric entries are converted to NaN, and completely
    empty rows or columns are removed.

    Parameters
    ----------
    path : Path
        Path to the ImageJ-generated CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned fluorescence-intensity dataframe.
    """

    # Read the CSV file
    df = pd.read_csv(path)

    # Convert all entries to numeric values.
    # Non-numeric entries are replaced with NaN.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove rows containing no usable measurements
    df = df.dropna(axis=0, how="all")

    # Remove columns containing no usable measurements
    df = df.dropna(axis=1, how="all")

    return df


# ============================================================
# ANALYSE A SINGLE CELL TRACE
# ============================================================

def analyse_cell_trace(
    trace,
    prominence=10,
    min_distance=1,
    frame_interval_seconds=30
):
    """
    Analyse the fluorescence trace of a single cell.

    Calcium peaks are identified using scipy.signal.find_peaks().
    A cell is classified as pulsatile when at least one peak is
    detected.

    Parameters
    ----------
    trace : array-like
        Fluorescence intensity measurements across successive frames.

    prominence : float
        Minimum peak prominence required for a signal to be classified
        as a calcium peak. This controls how strongly a peak must stand
        out from its surrounding fluorescence signal.

    min_distance : int
        Minimum number of frames separating detected peaks.

    frame_interval_seconds : float
        Time between successive image frames in seconds.

    Returns
    -------
    dict
        Summary statistics describing the fluorescence trace,
        detected peaks and pulse frequency.
    """

    # Convert trace to a NumPy array
    trace = np.asarray(trace, dtype=float)

    # Remove missing or infinite values
    trace = trace[np.isfinite(trace)]

    # --------------------------------------------------------
    # Basic fluorescence statistics
    # --------------------------------------------------------

    mean_val = float(np.mean(trace))
    std_val = float(np.std(trace))

    # Coefficient of variation:
    # variation in fluorescence relative to mean fluorescence
    cv = std_val / mean_val if mean_val != 0 else np.nan

    # --------------------------------------------------------
    # Calcium peak detection
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        trace,
        prominence=prominence,
        distance=min_distance
    )

    # Extract fluorescence values at detected peak positions
    peak_values = trace[peaks] if len(peaks) > 0 else np.array([])

    # --------------------------------------------------------
    # Calculate recording duration
    # --------------------------------------------------------

    duration_seconds = len(trace) * frame_interval_seconds
    duration_minutes = duration_seconds / 60

    # Total number of detected calcium peaks
    peak_count = int(len(peaks))

    # --------------------------------------------------------
    # Calculate pulse frequency
    # --------------------------------------------------------

    # Number of detected peaks per minute
    pulse_frequency_per_min = (
        peak_count / duration_minutes
        if duration_minutes > 0
        else np.nan
    )

    # Number of detected peaks per recorded image frame
    pulse_frequency_per_frame = (
        peak_count / len(trace)
        if len(trace) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Return summary statistics for this cell
    # --------------------------------------------------------

    return {
        "mean_intensity": mean_val,

        "max_intensity": float(np.max(trace)),

        "min_intensity": float(np.min(trace)),

        "std_intensity": std_val,

        "cv": float(cv),

        # Area under the fluorescence-intensity curve
        "auc": float(np.trapezoid(trace)),

        # Number of detected calcium peaks
        "peak_count": peak_count,

        # Mean fluorescence intensity at detected peaks
        "peak_amplitude_mean": (
            float(np.mean(peak_values))
            if len(peak_values) > 0
            else np.nan
        ),

        # Maximum fluorescence intensity among detected peaks
        "peak_amplitude_max": (
            float(np.max(peak_values))
            if len(peak_values) > 0
            else np.nan
        ),

        # Cell is classified as pulsatile if at least one peak
        # was detected during the recording
        "pulsing": len(peaks) >= 1,

        # Pulse frequency expressed as peaks per minute
        "pulse_frequency_per_min": float(
            pulse_frequency_per_min
        ),

        # Pulse frequency expressed as peaks per image frame
        "pulse_frequency_per_frame": float(
            pulse_frequency_per_frame
        )
    }


# ============================================================
# ADD NUMERICAL LABELS ABOVE BAR-PLOT BARS
# ============================================================

def add_bar_labels(ax, bars):
    """
    Add the numerical height of each bar above a bar plot.
    """

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom"
        )


# ============================================================
# ANALYSE ONE CALCIUM-IMAGING CSV FILE
# ============================================================

def analyse_one_csv(
    csv_path: Path,
    output_dir: Path,
    prominence: float = 10,
    min_distance: int = 1,
    n_example_traces: int = 30,
    frame_interval_seconds: float = 30
):
    """
    Analyse all individual-cell fluorescence traces contained
    within one calcium-imaging CSV file.

    For every cell, the script:
        1. extracts the fluorescence trace;
        2. detects calcium peaks;
        3. classifies the cell as pulsing or non-pulsing;
        4. calculates pulse frequency and fluorescence statistics.

    The function then calculates condition-level summary statistics
    and generates several quality-control and descriptive plots.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV file.

    output_dir : Path
        Directory in which analysis outputs will be saved.

    prominence : float
        Minimum prominence used by scipy.signal.find_peaks().

    min_distance : int
        Minimum separation between detected peaks, in frames.

    n_example_traces : int
        Number of individual-cell traces plotted for visualisation.

    frame_interval_seconds : float
        Time interval between consecutive image frames.

    Returns
    -------
    stats_df : pd.DataFrame
        Cell-level statistics.

    summary_df : pd.DataFrame
        Condition-level summary statistics.

    raw_df : pd.DataFrame
        Original cleaned fluorescence-intensity data.
    """

    # Use the input file name as the condition name
    condition_name = csv_path.stem

    # Create output directory if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fluorescence measurements
    raw_df = load_calcium_csv(csv_path)

    print("Loaded:", csv_path)
    print("Shape:", raw_df.shape)
    print("Rows = frames, columns = cells")

    # Store statistics from each individual cell
    rows = []

    # ========================================================
    # CELL-BY-CELL PEAK ANALYSIS
    # ========================================================

    for cell_index, col in enumerate(
        raw_df.columns,
        start=1
    ):

        # Extract fluorescence trace for one cell
        trace = raw_df[col].to_numpy(dtype=float)

        # Detect calcium peaks and calculate summary statistics
        stats = analyse_cell_trace(
            trace,
            prominence=prominence,
            min_distance=min_distance,
            frame_interval_seconds=frame_interval_seconds
        )

        # Add metadata identifying this cell and condition
        stats["condition"] = condition_name
        stats["cell_id"] = cell_index
        stats["column_name"] = col

        rows.append(stats)

    # Convert all individual-cell results into a dataframe
    stats_df = pd.DataFrame(rows)

    # ========================================================
    # CALCULATE PULSING CELL NUMBERS
    # ========================================================

    # Total number of analysed cells
    n_total = len(stats_df)

    # Number containing at least one detected calcium peak
    n_pulsing = int(stats_df["pulsing"].sum())

    # Number containing no detected peaks
    n_non_pulsing = n_total - n_pulsing

    # Percentage of cells classified as pulsatile
    percent_pulsing = 100 * n_pulsing / n_total

    # ========================================================
    # CONDITION-LEVEL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame([{

        "condition": condition_name,

        "n_total": n_total,

        "n_pulsing": n_pulsing,

        "n_non_pulsing": n_non_pulsing,

        "percent_pulsing": percent_pulsing,

        # Mean fluorescence intensity across cells
        "mean_intensity":
            stats_df["mean_intensity"].mean(),

        # Mean maximum fluorescence intensity across cells
        "mean_max_intensity":
            stats_df["max_intensity"].mean(),

        # Mean coefficient of variation
        "mean_cv":
            stats_df["cv"].mean(),

        # Mean area under the fluorescence curve
        "mean_auc":
            stats_df["auc"].mean(),

        # Average number of detected peaks across ALL cells,
        # including cells containing zero peaks
        "mean_peak_count":
            stats_df["peak_count"].mean(),

        # Mean pulse frequency across ALL cells.
        # Non-pulsing cells contribute a frequency of zero.
        "mean_pulse_frequency_per_min":
            stats_df["pulse_frequency_per_min"].mean(),

        "median_pulse_frequency_per_min":
            stats_df["pulse_frequency_per_min"].median(),

        # Mean pulse frequency calculated ONLY among cells
        # classified as pulsatile
        "mean_pulse_frequency_pulsing_cells_only":
            stats_df.loc[
                stats_df["pulsing"],
                "pulse_frequency_per_min"
            ].mean(),

    }])

    # ========================================================
    # SAVE NUMERICAL OUTPUTS
    # ========================================================

    # Save statistics for every individual cell
    stats_df.to_csv(
        output_dir / f"{condition_name}_cell_statistics.csv",
        index=False
    )

    # Save one-row summary of the experimental condition
    summary_df.to_csv(
        output_dir / f"{condition_name}_summary.csv",
        index=False
    )

    # ========================================================
    # PLOT 1:
    # NUMBER OF PULSING VS NON-PULSING CELLS
    # ========================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["Non-pulsing", "Pulsing"],
        [n_non_pulsing, n_pulsing]
    )

    # Display cell counts above bars
    add_bar_labels(ax, bars)

    ax.set_ylabel("Number of cells")

    ax.set_title(
        f"{condition_name}: "
        f"{n_pulsing}/{n_total} pulsing "
        f"({percent_pulsing:.1f}%)"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{condition_name}_pulsing_vs_non_pulsing.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 2:
    # REPRESENTATIVE FLUORESCENCE TRACES
    # ========================================================

    plt.figure(figsize=(10, 6))

    # Plot the first n_example_traces cells
    for col in raw_df.columns[:n_example_traces]:

        plt.plot(
            raw_df.index + 1,
            raw_df[col],
            linewidth=1
        )

    plt.xlabel("Frame")
    plt.ylabel("Intensity")

    plt.title(
        f"Example calcium traces: {condition_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{condition_name}_example_traces.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 3:
    # DISTRIBUTION OF PEAK COUNTS
    # ========================================================

    peak_counts = (
        stats_df["peak_count"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        peak_counts.index.astype(str),
        peak_counts.values
    )

    add_bar_labels(ax, bars)

    ax.set_xlabel("Peak count")
    ax.set_ylabel("Number of cells")
    ax.set_title("Peak count distribution")

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{condition_name}_peak_count_distribution.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 4:
    # COEFFICIENT OF VARIATION DISTRIBUTION
    # ========================================================

    counts, bins, patches = plt.hist(
        stats_df["cv"].dropna(),
        bins=30
    )

    # Add count labels above histogram bins
    for count, patch in zip(counts, patches):

        if count > 0:

            plt.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=7
            )

    plt.xlabel("Coefficient of variation")
    plt.ylabel("Number of cells")
    plt.title("CV distribution")

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{condition_name}_cv_distribution.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PLOT 5:
    # PULSE-FREQUENCY DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(7, 5))

    plt.hist(
        stats_df["pulse_frequency_per_min"].dropna(),
        bins=20
    )

    plt.xlabel("Pulse frequency (peaks/min)")
    plt.ylabel("Number of cells")
    plt.title("Pulse frequency distribution")

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{condition_name}_pulse_frequency_distribution.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT SUMMARY TO CONSOLE
    # ========================================================

    print("\nSummary:")
    print(summary_df)

    print("\nSaved to:")
    print(output_dir)

    return stats_df, summary_df, raw_df


# ============================================================
# RUN ANALYSIS
# ============================================================

if __name__ == "__main__":

    # Path to fluorescence measurements exported from ImageJ
    csv_path = Path(
        r"D:\Tarmizi\7 July\Control4_1\Results.csv"
    )

    # Directory in which outputs will be saved
    output_dir = Path(
        r"D:\Tarmizi\7 July\Control4_1"
    )

    # Run analysis for this experimental condition
    stats_df, summary_df, raw_df = analyse_one_csv(

        csv_path=csv_path,

        output_dir=output_dir,

        # A prominence of 20 was used for the 7 July dataset
        # because this dataset showed greater background noise.
        prominence=20,

        # Peaks can occur in immediately adjacent frames
        min_distance=1,

        # Number of traces displayed in the example-trace plot
        n_example_traces=30,

        # Images were acquired every 30 seconds
        frame_interval_seconds=30
    )