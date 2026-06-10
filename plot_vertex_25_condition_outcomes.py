#!/usr/bin/env python3
"""Plot condition matrix plus stacked outcome bars for vertex-25 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTENUATED = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
SIB_COMP = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
MAT_AGE = [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
MAT_MORTALITY = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
EPIGENETIC = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]

CONDITION_NAMES = [
    "Attenuated",
    "Sibling\ncompetition",
    "Maternal\nage",
    "Maternal\nmortality",
    "Epigenetic",
]

STATUS_ORDER = ["extinct", "succeed", "failed"]
STATUS_COLORS = {
    "extinct": "#313695",
    "succeed": "#006d3c",
    "failed": "#c51b2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a condition/outcome stacked-bar figure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out/out_vertex_25_summary.txt"),
        help="Tab-delimited summary table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/vertex_25_condition_outcomes.png"),
        help="Output figure path. Extension controls format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster outputs.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", na_values=["NA"])
    needed = {
        "sib_mortality",
        "maternal_age_effect",
        "mat_mortality",
        "attenuation_cutoff",
        "if_epi",
        "status",
    }
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def rows_for_condition(
    df: pd.DataFrame,
    attenuated: int,
    sib_comp: int,
    mat_age: int,
    mat_mortality: int,
    epigenetic: int,
) -> pd.DataFrame:
    base = (
        (df["sib_mortality"] == sib_comp)
        & (df["maternal_age_effect"] == mat_age)
        & (df["mat_mortality"] == mat_mortality)
        & (df["if_epi"] == epigenetic)
    )
    cutoff = 0.1 if attenuated else 0.0
    rows = df.loc[base & (df["attenuation_cutoff"] == cutoff)]

    # The all-effects-off baseline has no attenuation parameter in the file.
    if rows.empty and sib_comp == mat_age == mat_mortality == epigenetic == 0:
        rows = df.loc[base & df["attenuation_cutoff"].isna()]

    return rows


def build_plot_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for i, condition in enumerate(
        zip(ATTENUATED, SIB_COMP, MAT_AGE, MAT_MORTALITY, EPIGENETIC), start=1
    ):
        rows = rows_for_condition(df, *condition)
        if rows.empty:
            raise ValueError(f"No rows matched requested condition {i}: {condition}")

        counts = rows["status"].value_counts()
        record = {
            "condition_id": i,
            "n": int(counts.sum()),
            "attenuated": condition[0],
            "sib_comp": condition[1],
            "mat_age": condition[2],
            "mat_mortality": condition[3],
            "epigenetic": condition[4],
        }
        for status in STATUS_ORDER:
            record[status] = int(counts.get(status, 0))
            record[f"{status}_prop"] = record[status] / record["n"]
        records.append(record)

    return pd.DataFrame.from_records(records)


def add_condition_matrix(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
    matrix = plot_df[
        ["attenuated", "sib_comp", "mat_age", "mat_mortality", "epigenetic"]
    ].to_numpy()
    y_positions = np.arange(len(plot_df))[::-1]
    x_positions = np.arange(matrix.shape[1])

    for row_idx, y in enumerate(y_positions):
        for col_idx, x in enumerate(x_positions):
            filled = bool(matrix[row_idx, col_idx])
            ax.scatter(
                x,
                y,
                s=180,
                facecolors="black" if filled else "white",
                edgecolors="black",
                linewidths=1.6,
                zorder=3,
            )

    for x, label in zip(x_positions, CONDITION_NAMES):
        ax.text(
            x,
            y_positions[0] + 0.58,
            label,
            ha="left",
            va="bottom",
            rotation=45,
            fontsize=9,
        )

    ax.text(
        -1.65,
        y_positions[0] + 0.58,
        "Conditions:",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    ax.set_xlim(-1.75, matrix.shape[1] - 0.25)
    ax.set_ylim(-0.8, len(plot_df) - 0.2)
    ax.axis("off")


def add_outcome_bars(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
    y_positions = np.arange(len(plot_df))[::-1]
    left = np.zeros(len(plot_df))

    for status in STATUS_ORDER:
        values = plot_df[f"{status}_prop"].to_numpy()
        ax.barh(
            y_positions,
            values,
            left=left,
            height=0.58,
            color=STATUS_COLORS[status],
            edgecolor="none",
            label=status.capitalize(),
        )
        left += values

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(plot_df) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Outcome proportion", fontsize=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(axis="x", length=4, width=1)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(STATUS_ORDER),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
    )


def make_figure(plot_df: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11, 8.2), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 2.45],
        left=0.07,
        right=0.98,
        top=0.9,
        bottom=0.11,
        wspace=0.02,
    )
    matrix_ax = fig.add_subplot(grid[0, 0])
    bars_ax = fig.add_subplot(grid[0, 1])

    add_condition_matrix(matrix_ax, plot_df)
    add_outcome_bars(bars_ax, plot_df)
    return fig


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input)
    plot_df = build_plot_table(summary)
    fig = make_figure(plot_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
