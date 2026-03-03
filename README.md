# Parkinson's Disease Drug Discovery Pipeline
## Molecular Docking Targeting GBA (Glucocerebrosidase) Mutations

A computational drug discovery pipeline for identifying small molecule therapeutics
that target GBA gene mutations — the most common genetic risk factor for Parkinson's disease.

## Scientific Background

### The GBA-Parkinson's Connection

Mutations in the **GBA gene** (encoding the lysosomal enzyme glucocerebrosidase / GCase)
are the most significant genetic risk factor for Parkinson's disease (PD):

- **5-10%** of PD patients carry a GBA mutation
- **N370S** mutation increases PD risk **5.4x** (most common)
- **L444P** mutation increases PD risk **9.1x** (causes protein misfolding)

### How GBA Mutations Cause Parkinson's

```
GBA Mutation → Misfolded GCase → ER retention → Reduced lysosomal GCase
                                                        ↓
                                            ↓ Glucosylceramide accumulation
                                            ↓ Impaired autophagy-lysosome pathway
                                            ↓ α-Synuclein accumulation & aggregation
                                            ↓ Dopaminergic neuron death
                                                        ↓
                                                Parkinson's Disease
```

### Therapeutic Strategy

This pipeline screens compounds using three strategies:

1. **Pharmacological Chaperones**: Bind mutant GCase in the ER, stabilize its
   folding, and enhance trafficking to lysosomes (e.g., Ambroxol, Isofagomine)
2. **Small Molecule Activators**: Allosterically boost GCase enzymatic activity
   (e.g., LTI-291, S-181)
3. **Substrate Reduction**: Reduce glucosylceramide accumulation downstream
   (e.g., Venglustat)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. TARGET PREPARATION (gba_target.py)                      │
│     ├── Fetch GBA/GCase PDB structure (2NT0, 2NT1)          │
│     ├── Clean: remove water, add hydrogens                  │
│     ├── Define active site / allosteric binding box          │
│     └── Generate PDBQT for docking                          │
│                                                             │
│  2. COMPOUND LIBRARY (compound_library.py)                  │
│     ├── Known GCase modulators (Ambroxol, NCGC607, etc.)    │
│     ├── Computationally designed candidates                 │
│     ├── Reference inhibitors for validation                 │
│     ├── Lipinski Rule of Five filter                        │
│     └── BBB permeability filter                             │
│                                                             │
│  3. MOLECULAR DOCKING (docking_engine.py)                   │
│     ├── AutoDock Vina scoring (if installed)                 │
│     ├── Built-in empirical scoring (fallback)               │
│     ├── VdW + H-bond + electrostatic + desolvation          │
│     └── Multi-pose generation & ranking                     │
│                                                             │
│  4. ADMET FILTERING (admet_filter.py)                       │
│     ├── BBB penetration prediction (critical for PD)        │
│     ├── CYP450 inhibition (DDI with PD medications)         │
│     ├── hERG cardiac safety                                 │
│     ├── Hepatotoxicity & mutagenicity                       │
│     └── Oral bioavailability & half-life                    │
│                                                             │
│  5. ANALYSIS & REPORTING (analysis.py)                      │
│     ├── Composite ranking (docking + ADMET + BBB)           │
│     ├── Binding affinity bar charts                         │
│     ├── ADMET radar plots                                   │
│     ├── SAR scatter plots                                   │
│     └── Lead compound recommendation                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd parkinsons-molecular-docking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install AutoDock Vina for production docking
```bash
pip install vina
```

> The pipeline includes a built-in empirical scoring function and works
> without Vina installed. For publication-quality results, install Vina.

## Usage

### Basic: Screen against N370S mutation (most common)
```bash
python -m parkinsons_docking.pipeline --mutation N370S
```

### Target the severe L444P mutation
```bash
python -m parkinsons_docking.pipeline --mutation L444P
```

### Screen allosteric chaperone binding site
```bash
python -m parkinsons_docking.pipeline --mutation N370S --site allosteric
```

### High-accuracy docking (slower)
```bash
python -m parkinsons_docking.pipeline --mutation N370S --exhaustiveness 64
```

### Screen all GBA-PD mutations
```bash
python -m parkinsons_docking.pipeline --all-mutations
```

### Disable BBB filter (include all compounds)
```bash
python -m parkinsons_docking.pipeline --no-bbb-filter
```

## Compound Library

### Known GCase Modulators
| Compound | Category | Status | Mechanism |
|----------|----------|--------|-----------|
| Ambroxol | Chaperone | Clinical (Phase 2) | pH-dependent inhibitory chaperone |
| Isofagomine | Chaperone | Clinical (discontinued) | Iminosugar transition-state mimic |
| NCGC607 | Non-inhibitory chaperone | Preclinical | Stabilizes without active site competition |
| LTI-291 | Activator | Clinical (Phase 1) | Allosteric GCase activator |
| S-181 | Activator | Preclinical | Brain-penetrant GCase modulator |
| Venglustat | Substrate reduction | Clinical (Phase 2) | GlcCer synthase inhibitor |

### GBA Mutations Targeted
| Mutation | PD Risk | Severity | Effect |
|----------|---------|----------|--------|
| N370S | 5.4x | Mild | Reduced catalytic activity |
| L444P | 9.1x | Severe | Protein misfolding, ER retention |
| E326K | 1.7x | Mild | Reduced enzyme stability |
| T369M | 2.4x | Mild | Affects substrate positioning |
| D409H | 11.0x | Severe | Destabilizes protein fold |

## Output Files

After running the pipeline, the `results/` directory contains:

```
results/
├── structures/          # Prepared PDB and PDBQT files
│   ├── 2NT0.pdb         # Wild-type GCase structure
│   ├── 2NT0.clean.pdbqt # Prepared receptor
│   └── *.pdbqt          # Prepared ligands
├── docking_results.json # Raw docking scores & poses
├── final_rankings.json  # Composite-ranked compounds
├── docking_report.txt   # Full text report
├── binding_affinities.png # Binding affinity bar chart
├── admet_radar.png      # ADMET property radar plot
└── sar_scatter.png      # Structure-Activity Relationship plot
```

## Project Structure

```
parkinsons_docking/
├── __init__.py           # Package info & scientific context
├── gba_target.py         # GBA protein target preparation
├── compound_library.py   # Drug candidate library & filtering
├── docking_engine.py     # Molecular docking simulations
├── admet_filter.py       # ADMET property prediction & filtering
├── analysis.py           # Results analysis & visualization
├── pipeline.py           # Main pipeline orchestrator
└── utils/                # Utility functions
```

## References

1. Sidransky E, et al. "Multicenter analysis of glucocerebrosidase mutations in Parkinson's disease." *N Engl J Med.* 2009;361(17):1651-1661.
2. Maegawa GH, et al. "Identification and characterization of ambroxol as an enzyme enhancement agent for Gaucher disease." *J Biol Chem.* 2009;284(35):23502-23516.
3. Mazzulli JR, et al. "Activation of β-Glucocerebrosidase Reduces Pathological α-Synuclein and Restores Lysosomal Function in Parkinson's Patient Midbrain Neurons." *J Neurosci.* 2016;36(29):7693-7706.
4. Balestrino R, Bhf Schapira A. "Glucocerebrosidase and Parkinson Disease: Molecular, Clinical, and Therapeutic Implications." *Neuroscientist.* 2018;24(5):540-559.
5. Trott O, Olson AJ. "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." *J Comput Chem.* 2010;31(2):455-461.
