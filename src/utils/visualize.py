"""Visualization utilities for optimization results."""

import os
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")


class VisualizationReport:
    """Generate PDF reports of optimization results."""

    def __init__(
        self,
        filename: str = "visualization_results.pdf",
        output_dir: str = ".",
    ) -> None:
        self.filename = filename
        self.output_dir = output_dir
        self.groups: list[list] = []
        self.current_group: list = []

    def add_chart_to_group(self, chart_func, title: str | None = None) -> None:
        self.current_group.append(chart_func)

    def finalize_group(self) -> None:
        if self.current_group:
            self.groups.append(self.current_group)
        self.current_group = []

    def generate_pdf(self) -> str:
        """Generate the PDF report and return the file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, self.filename)

        with PdfPages(output_file) as pdf:
            for group in self.groups:
                for chart_func in group:
                    fig = plt.figure()
                    chart_func()
                    pdf.savefig(fig)
                    plt.close(fig)

        return output_file


def visualize_result(
    result: Any,
    parameters: Any,
    output_path: str = "optimization_result.pdf",
) -> Optional[str]:
    """Generate a visualization PDF from optimization results.

    Args:
        result: SimulationResult or GeneticSolution
        parameters: GeneticOptimizationParameters
        output_path: Output file path

    Returns:
        Path to the generated PDF, or None if visualization failed.
    """
    try:
        report = VisualizationReport(
            filename=os.path.basename(output_path),
            output_dir=os.path.dirname(output_path) or ".",
        )

        def overview_chart():
            sim = (
                result
                if hasattr(result, "costs_per_dt")
                else getattr(result, "result", None)
            )
            if sim is None:
                return

            n = len(sim.costs_per_dt)
            x = list(range(n))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax1.plot(
                x,
                list(sim.grid_import_wh_per_dt),
                label="Grid Import (Wh)",
                color="red",
            )
            ax1.plot(
                x,
                list(sim.self_consumption_wh_per_dt),
                label="Self-consumption (Wh)",
                color="green",
            )
            ax1.plot(x, list(sim.feedin_wh_per_dt), label="Feed-in (Wh)", color="blue")
            ax1.plot(x, list(sim.losses_wh_per_dt), label="Losses (Wh)", color="orange")
            ax1.set_ylabel("Energy (Wh)")
            ax1.set_title("Energy Flows")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(
                x,
                list(sim.costs_per_dt),
                label="Costs (€)",
                color="red",
                linestyle="--",
            )
            ax2.plot(
                x,
                list(sim.revenue_per_dt),
                label="Revenue (€)",
                color="green",
                linestyle="--",
            )
            ax2.set_ylabel("Euro")
            ax2.set_xlabel("Hour")
            ax2.set_title(f"Financial (Total: {sim.net_balance:.4f} €)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

        report.add_chart_to_group(overview_chart, "Overview")
        report.finalize_group()
        return report.generate_pdf()

    except Exception as ex:
        print(f"Visualization failed: {ex}")
        return None
