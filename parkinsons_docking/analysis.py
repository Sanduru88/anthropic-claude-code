"""
Results Analysis & Visualization for GBA Docking Campaign
============================================================
Generates comprehensive analysis reports including:
- Binding affinity rankings and distributions
- ADMET property comparisons
- Structure-activity relationship (SAR) analysis
- Publication-ready figures
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DockingAnalyzer:
    """
    Analyzes and visualizes molecular docking results for GBA drug discovery.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def rank_compounds(self, docking_results: list, admet_profiles: list) -> list[dict]:
        """
        Rank compounds by combined docking score and ADMET profile.

        Composite Score = 0.5 * normalized_docking + 0.3 * ADMET_score + 0.2 * drug_likeness

        Args:
            docking_results: List of DockingResult objects
            admet_profiles: List of ADMETProfile objects

        Returns:
            Ranked list of compound dictionaries
        """
        admet_map = {p.compound_name: p for p in admet_profiles}

        rankings = []
        affinities = [r.binding_affinity for r in docking_results]
        min_aff = min(affinities) if affinities else -10
        max_aff = max(affinities) if affinities else -3
        aff_range = max_aff - min_aff if max_aff != min_aff else 1.0

        for result in docking_results:
            admet = admet_map.get(result.compound_name)

            # Normalize docking score (0-1, higher = better binder)
            norm_docking = (max_aff - result.binding_affinity) / aff_range

            # ADMET score
            admet_score = admet.overall_score if admet else 0.5

            # Composite score
            composite = 0.5 * norm_docking + 0.3 * admet_score + 0.2 * (1.0 if admet and admet.bbb_penetration else 0.0)

            entry = {
                "rank": 0,
                "compound_name": result.compound_name,
                "binding_affinity": result.binding_affinity,
                "binding_category": result.binding_category(),
                "admet_score": round(admet_score, 3),
                "bbb_penetrant": admet.bbb_penetration if admet else False,
                "admet_pass": admet.passes_all_filters if admet else False,
                "composite_score": round(composite, 3),
                "flags": admet.flags if admet else [],
                "interactions": result.interactions,
            }
            rankings.append(entry)

        rankings.sort(key=lambda x: -x["composite_score"])
        for i, entry in enumerate(rankings, 1):
            entry["rank"] = i

        return rankings

    def generate_report(self, rankings: list, target_name: str,
                        mutation: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report of the docking campaign.

        Args:
            rankings: Ranked compound list from rank_compounds()
            target_name: Name of protein target
            mutation: GBA mutation targeted

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "MOLECULAR DOCKING REPORT",
            f"Parkinson's Disease Drug Discovery - GBA/GCase Target",
            "=" * 80,
            "",
            f"Target: {target_name}",
            f"Mutation: {mutation or 'Wild-type'}",
            f"Compounds Screened: {len(rankings)}",
            "",
        ]

        if mutation:
            lines.extend([
                "Therapeutic Strategy:",
                f"  Targeting GBA {mutation} mutation to restore GCase function.",
                "  Pharmacological chaperones stabilize mutant enzyme folding,",
                "  enhance lysosomal trafficking, and reduce alpha-synuclein",
                "  accumulation in dopaminergic neurons.",
                "",
            ])

        # Top hits table
        lines.extend([
            "-" * 80,
            "TOP RANKED COMPOUNDS",
            "-" * 80,
            f"{'Rank':<5} {'Compound':<25} {'Affinity':>10} {'Category':>10} "
            f"{'ADMET':>7} {'BBB':>5} {'Score':>7}",
            "-" * 80,
        ])

        for entry in rankings:
            lines.append(
                f"{entry['rank']:<5} "
                f"{entry['compound_name']:<25} "
                f"{entry['binding_affinity']:>9.2f} "
                f"{entry['binding_category']:>10} "
                f"{'PASS' if entry['admet_pass'] else 'FAIL':>7} "
                f"{'Yes' if entry['bbb_penetrant'] else 'No':>5} "
                f"{entry['composite_score']:>7.3f}"
            )

        # Detailed analysis of top 5
        lines.extend(["", "-" * 80, "DETAILED ANALYSIS - TOP 5 CANDIDATES", "-" * 80])

        for entry in rankings[:5]:
            lines.extend([
                "",
                f"  #{entry['rank']}: {entry['compound_name']}",
                f"    Binding Affinity: {entry['binding_affinity']:.2f} kcal/mol ({entry['binding_category']})",
                f"    Composite Score:  {entry['composite_score']:.3f}",
                f"    ADMET Score:      {entry['admet_score']:.3f}",
                f"    BBB Penetrant:    {'Yes' if entry['bbb_penetrant'] else 'No'}",
                f"    ADMET Pass:       {'Yes' if entry['admet_pass'] else 'No'}",
            ])

            if entry["flags"]:
                lines.append("    Flags:")
                for flag in entry["flags"]:
                    lines.append(f"      - {flag}")

            if entry["interactions"]:
                lines.append("    Scoring Components:")
                for k, v in entry["interactions"].items():
                    lines.append(f"      {k}: {v}")

        # Summary statistics
        affinities = [e["binding_affinity"] for e in rankings]
        passing = [e for e in rankings if e["admet_pass"]]
        bbb_yes = [e for e in rankings if e["bbb_penetrant"]]

        lines.extend([
            "",
            "-" * 80,
            "SUMMARY STATISTICS",
            "-" * 80,
            f"  Total compounds screened:  {len(rankings)}",
            f"  Mean binding affinity:     {np.mean(affinities):.2f} kcal/mol",
            f"  Best binding affinity:     {min(affinities):.2f} kcal/mol",
            f"  Excellent binders (<-9.0): {sum(1 for a in affinities if a < -9.0)}",
            f"  Good binders (<-7.0):      {sum(1 for a in affinities if a < -7.0)}",
            f"  ADMET passing:             {len(passing)}",
            f"  BBB penetrant:             {len(bbb_yes)}",
            f"  Viable candidates:         {sum(1 for e in rankings if e['admet_pass'] and e['binding_affinity'] < -7.0)}",
            "",
        ])

        if passing:
            best_viable = passing[0]
            lines.extend([
                "-" * 80,
                "LEAD COMPOUND RECOMMENDATION",
                "-" * 80,
                f"  Recommended Lead: {best_viable['compound_name']}",
                f"  Binding Affinity: {best_viable['binding_affinity']:.2f} kcal/mol",
                f"  Composite Score:  {best_viable['composite_score']:.3f}",
                "",
                "  Next Steps:",
                "  1. Validate binding pose with MD simulation (100ns)",
                "  2. Compute binding free energy (MM-PBSA/GBSA)",
                "  3. Synthesize top 3-5 candidates",
                "  4. In vitro GCase activity assay (4-MU-Glc substrate)",
                "  5. Cell-based assay in GBA-mutant fibroblasts",
                "  6. iPSC-derived dopaminergic neuron assay",
                "  7. In vivo pharmacokinetics (mouse)",
                "  8. GBA-PD mouse model efficacy study",
                "",
            ])

        lines.append("=" * 80)

        report = "\n".join(lines)
        return report

    def plot_binding_affinities(self, rankings: list, output_file: Optional[str] = None):
        """Generate binding affinity bar chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(12, 6))

            names = [e["compound_name"] for e in rankings]
            affinities = [e["binding_affinity"] for e in rankings]
            colors = []
            for e in rankings:
                if e["admet_pass"] and e["binding_affinity"] < -7.0:
                    colors.append("#2ecc71")  # Green: viable lead
                elif e["binding_affinity"] < -7.0:
                    colors.append("#f39c12")  # Orange: good binder, ADMET issues
                else:
                    colors.append("#e74c3c")  # Red: weak binder

            bars = ax.barh(range(len(names)), affinities, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Binding Affinity (kcal/mol)", fontsize=12)
            ax.set_title(
                "GBA/GCase Molecular Docking Results\n"
                "Parkinson's Disease Drug Discovery Campaign",
                fontsize=14,
            )
            ax.axvline(x=-7.0, color="gray", linestyle="--", alpha=0.7, label="Good binder threshold")
            ax.axvline(x=-9.0, color="gray", linestyle=":", alpha=0.7, label="Excellent binder threshold")
            ax.legend()
            ax.invert_yaxis()

            plt.tight_layout()

            if output_file is None:
                output_file = str(self.output_dir / "binding_affinities.png")
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Binding affinity plot saved to {output_file}")

        except ImportError:
            logger.warning("matplotlib/seaborn not available - skipping plot generation")

    def plot_admet_radar(self, rankings: list, top_n: int = 5,
                         output_file: Optional[str] = None):
        """Generate ADMET radar/spider chart for top compounds."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            categories = ["Docking\nScore", "BBB\nScore", "ADMET\nScore",
                          "Bioavail.", "Safety"]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            for entry in rankings[:top_n]:
                values = [
                    min(abs(entry["binding_affinity"]) / 12.0, 1.0),
                    1.0 if entry["bbb_penetrant"] else 0.3,
                    entry["admet_score"],
                    0.8 if entry["admet_pass"] else 0.3,
                    0.9 if not entry["flags"] else max(0.2, 1.0 - len(entry["flags"]) * 0.15),
                ]
                values += values[:1]
                ax.plot(angles, values, "o-", linewidth=2, label=entry["compound_name"])
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title("Multi-Parameter Optimization\nTop GBA Drug Candidates", fontsize=14, pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

            plt.tight_layout()
            if output_file is None:
                output_file = str(self.output_dir / "admet_radar.png")
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"ADMET radar plot saved to {output_file}")

        except ImportError:
            logger.warning("matplotlib not available - skipping radar plot")

    def plot_sar_scatter(self, rankings: list, compounds: list,
                         output_file: Optional[str] = None):
        """Generate SAR scatter plot: LogP vs MW colored by binding affinity."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            compound_map = {c.name: c for c in compounds}

            fig, ax = plt.subplots(figsize=(10, 7))

            mws = []
            logps = []
            affinities = []
            names = []

            for entry in rankings:
                comp = compound_map.get(entry["compound_name"])
                if comp:
                    mws.append(comp.mol_weight)
                    logps.append(comp.logp)
                    affinities.append(entry["binding_affinity"])
                    names.append(entry["compound_name"])

            scatter = ax.scatter(mws, logps, c=affinities, cmap="RdYlGn_r",
                                 s=100, edgecolors="black", linewidth=0.5)

            for i, name in enumerate(names):
                ax.annotate(name, (mws[i], logps[i]), fontsize=7,
                            xytext=(5, 5), textcoords="offset points")

            # Drug-like property space box
            ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="LogP=5 (Lipinski)")
            ax.axvline(x=500, color="red", linestyle="--", alpha=0.5, label="MW=500 (Lipinski)")

            # BBB-favorable region
            from matplotlib.patches import Rectangle
            bbb_rect = Rectangle((150, 1.0), 300, 3.0, linewidth=2,
                                 edgecolor="blue", facecolor="blue", alpha=0.05)
            ax.add_patch(bbb_rect)
            ax.text(155, 3.8, "BBB-favorable\nregion", fontsize=8, color="blue", alpha=0.7)

            cbar = plt.colorbar(scatter, label="Binding Affinity (kcal/mol)")
            ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
            ax.set_ylabel("LogP", fontsize=12)
            ax.set_title("Structure-Activity Relationship\nGBA Drug Candidates", fontsize=14)
            ax.legend(fontsize=9)

            plt.tight_layout()
            if output_file is None:
                output_file = str(self.output_dir / "sar_scatter.png")
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SAR scatter plot saved to {output_file}")

        except ImportError:
            logger.warning("matplotlib not available - skipping SAR plot")

    def save_results_json(self, rankings: list, filename: str = "final_rankings.json"):
        """Save final rankings to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(rankings, f, indent=2, default=str)
        logger.info(f"Rankings saved to {output_path}")

    def save_report(self, report: str, filename: str = "docking_report.txt"):
        """Save text report to file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
