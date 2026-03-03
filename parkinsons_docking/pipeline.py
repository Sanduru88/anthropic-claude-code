"""
GBA-Parkinson's Disease Molecular Docking Pipeline
=====================================================
Main orchestrator that runs the complete drug discovery workflow:

1. Prepare GBA protein target (wild-type and/or mutant)
2. Load and filter compound library
3. Prepare ligands for docking
4. Run molecular docking simulations
5. Evaluate ADMET properties
6. Rank candidates by composite score
7. Generate analysis report and visualizations

Usage:
    python -m parkinsons_docking.pipeline
    python -m parkinsons_docking.pipeline --mutation N370S --site active_site
    python -m parkinsons_docking.pipeline --mutation L444P --bbb-filter
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from parkinsons_docking.gba_target import GBATargetPreparation, get_mutation_info, GBA_PD_MUTATIONS
from parkinsons_docking.compound_library import CompoundLibrary
from parkinsons_docking.docking_engine import DockingEngine
from parkinsons_docking.admet_filter import ADMETFilter
from parkinsons_docking.analysis import DockingAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(mutation: str = "N370S", site_type: str = "active_site",
                 bbb_filter: bool = True, exhaustiveness: int = 32,
                 output_dir: str = "results"):
    """
    Run the complete GBA molecular docking pipeline.

    Args:
        mutation: GBA mutation to target (N370S, L444P, E326K, T369M, D409H)
        site_type: 'active_site' or 'allosteric'
        bbb_filter: Apply blood-brain barrier filter
        exhaustiveness: Docking thoroughness (8-64, higher=slower+better)
        output_dir: Directory for output files
    """
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PARKINSON'S DISEASE DRUG DISCOVERY PIPELINE")
    print("  Targeting GBA (Glucocerebrosidase) Mutations")
    print("=" * 70)
    print()

    # ================================================================
    # STEP 1: Display mutation information
    # ================================================================
    print(f"[1/7] Target Information")
    print("-" * 40)
    if mutation:
        try:
            mut_info = get_mutation_info(mutation)
            print(f"  Mutation:   GBA {mutation}")
            print(f"  Position:   {mut_info['position']}")
            print(f"  Change:     {mut_info['wild_type']} -> {mut_info['mutant']}")
            print(f"  Severity:   {mut_info['severity']}")
            print(f"  PD Risk OR: {mut_info['pd_risk_or']}x")
            print(f"  Effect:     {mut_info['effect']}")
        except ValueError as e:
            print(f"  Warning: {e}")
            print(f"  Proceeding with wild-type GCase")
            mutation = None
    else:
        print("  Target: Wild-type GCase (no specific mutation)")
    print(f"  Binding site: {site_type}")
    print()

    # ================================================================
    # STEP 2: Prepare protein target
    # ================================================================
    print(f"[2/7] Preparing GBA Protein Target")
    print("-" * 40)
    target_prep = GBATargetPreparation(output_dir=str(output_path / "structures"))
    target = target_prep.prepare_target(mutation=mutation, site_type=site_type)
    print(f"  Target name:  {target.name}")
    print(f"  PDB ID:       {target.pdb_id}")
    print(f"  Structure:    {target.structure_path}")
    print(f"  PDBQT:        {target.pdbqt_path}")
    print(f"  Active site:  {', '.join(target.active_site_residues)}")
    print(f"  Binding box:  center=({target.binding_box['center']['x']:.1f}, "
          f"{target.binding_box['center']['y']:.1f}, "
          f"{target.binding_box['center']['z']:.1f})")
    print()

    # ================================================================
    # STEP 3: Load compound library
    # ================================================================
    print(f"[3/7] Loading Compound Library")
    print("-" * 40)
    library = CompoundLibrary()
    library.load_all()
    print(library.summary())
    print()

    # ================================================================
    # STEP 4: Filter and prepare compounds
    # ================================================================
    print(f"[4/7] Filtering & Preparing Compounds for Docking")
    print("-" * 40)
    screening_set = library.get_screening_set(
        include_reference=True,
        bbb_filter=bbb_filter,
    )
    print(f"  Screening set size: {len(screening_set)}")
    for c in screening_set:
        print(f"    - {c.name} ({c.category}, MW={c.mol_weight:.1f})")

    ligand_pdbqts = library.prepare_for_docking(
        compounds=screening_set,
        output_dir=str(output_path / "structures"),
    )
    print(f"  Prepared {len(ligand_pdbqts)} ligand PDBQT files")
    print()

    # ================================================================
    # STEP 5: Run molecular docking
    # ================================================================
    print(f"[5/7] Running Molecular Docking Simulations")
    print("-" * 40)
    engine = DockingEngine(
        exhaustiveness=exhaustiveness,
        n_poses=9,
        seed=42,
    )
    print(f"  Engine: {'AutoDock Vina' if engine.vina_available else 'Built-in empirical scoring'}")
    print(f"  Exhaustiveness: {exhaustiveness}")
    print(f"  CPUs: {engine.num_cpus}")
    print()

    docking_results = engine.dock_library(
        receptor_pdbqt=target.pdbqt_path,
        ligand_pdbqts=ligand_pdbqts,
        binding_box=target.binding_box,
        target_name=target.name,
        site_type=site_type,
    )

    engine.save_results(docking_results, str(output_path / "docking_results.json"))
    print()

    # ================================================================
    # STEP 6: ADMET evaluation
    # ================================================================
    print(f"[6/7] Evaluating ADMET Properties")
    print("-" * 40)
    admet_filter = ADMETFilter()
    admet_profiles = admet_filter.evaluate_library(screening_set)
    print()
    print(admet_filter.summary_table())
    print()

    # ================================================================
    # STEP 7: Analysis & reporting
    # ================================================================
    print(f"[7/7] Generating Analysis & Report")
    print("-" * 40)
    analyzer = DockingAnalyzer(output_dir=str(output_path))

    # Rank compounds
    rankings = analyzer.rank_compounds(docking_results, admet_profiles)

    # Generate report
    report = analyzer.generate_report(rankings, target.name, mutation)
    print(report)

    # Save outputs
    analyzer.save_results_json(rankings)
    analyzer.save_report(report)

    # Generate plots
    analyzer.plot_binding_affinities(rankings)
    analyzer.plot_admet_radar(rankings)
    analyzer.plot_sar_scatter(rankings, screening_set)

    # ================================================================
    # Pipeline complete
    # ================================================================
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Results directory: {output_path}")
    print(f"  Files generated:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            print(f"    - {f.relative_to(output_path)}")
    print("=" * 70)

    return rankings


def main():
    parser = argparse.ArgumentParser(
        description="GBA-Parkinson's Disease Molecular Docking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen against N370S mutation (most common GBA-PD mutation)
  python -m parkinsons_docking.pipeline --mutation N370S

  # Screen against L444P mutation (severe misfolding)
  python -m parkinsons_docking.pipeline --mutation L444P

  # Target allosteric site for chaperone discovery
  python -m parkinsons_docking.pipeline --mutation N370S --site allosteric

  # Wild-type GCase with high exhaustiveness
  python -m parkinsons_docking.pipeline --exhaustiveness 64

  # All mutations scan
  python -m parkinsons_docking.pipeline --all-mutations
        """,
    )

    parser.add_argument(
        "--mutation", type=str, default="N370S",
        choices=list(GBA_PD_MUTATIONS.keys()),
        help="GBA mutation to target (default: N370S)",
    )
    parser.add_argument(
        "--site", type=str, default="active_site",
        choices=["active_site", "allosteric"],
        help="Binding site type (default: active_site)",
    )
    parser.add_argument(
        "--bbb-filter", action="store_true", default=True,
        help="Apply blood-brain barrier permeability filter (default: True)",
    )
    parser.add_argument(
        "--no-bbb-filter", action="store_true",
        help="Disable BBB filter to include all compounds",
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=32,
        help="Docking exhaustiveness 8-64 (default: 32)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--all-mutations", action="store_true",
        help="Run pipeline against all known GBA-PD mutations",
    )

    args = parser.parse_args()

    bbb_filter = not args.no_bbb_filter

    if args.all_mutations:
        all_results = {}
        for mutation in GBA_PD_MUTATIONS:
            print(f"\n{'#' * 70}")
            print(f"# MUTATION: {mutation}")
            print(f"{'#' * 70}\n")
            results = run_pipeline(
                mutation=mutation,
                site_type=args.site,
                bbb_filter=bbb_filter,
                exhaustiveness=args.exhaustiveness,
                output_dir=f"{args.output_dir}/{mutation}",
            )
            all_results[mutation] = results

        # Cross-mutation comparison
        print("\n" + "=" * 70)
        print("CROSS-MUTATION COMPARISON")
        print("=" * 70)
        print(f"{'Mutation':<10} {'Best Compound':<25} {'Affinity':>10} {'Score':>8}")
        print("-" * 55)
        for mutation, rankings in all_results.items():
            if rankings:
                best = rankings[0]
                print(
                    f"{mutation:<10} {best['compound_name']:<25} "
                    f"{best['binding_affinity']:>9.2f} "
                    f"{best['composite_score']:>8.3f}"
                )
    else:
        run_pipeline(
            mutation=args.mutation,
            site_type=args.site,
            bbb_filter=bbb_filter,
            exhaustiveness=args.exhaustiveness,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
