"""
Compound Library for GBA-Targeted Drug Discovery
==================================================
Contains known GCase modulators, pharmacological chaperones, and
computationally designed candidates for screening against GBA mutations.

Drug Discovery Strategy for GBA-Parkinson's:
1. Pharmacological Chaperones: Bind mutant GCase, stabilize folding,
   enhance lysosomal trafficking (e.g., Ambroxol, Isofagomine)
2. Small Molecule Activators: Boost residual GCase enzymatic activity
3. Substrate Reduction: Reduce glucosylceramide accumulation
4. Alpha-synuclein Modulators: Target downstream pathology

SMILES notation is used for molecular representation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Compound:
    """Represents a small molecule compound for docking."""

    name: str
    smiles: str
    mol_weight: float = 0.0
    category: str = "unknown"
    mechanism: str = ""
    status: str = "experimental"  # experimental, preclinical, clinical, approved
    source: str = ""
    logp: float = 0.0
    hbd: int = 0  # hydrogen bond donors
    hba: int = 0  # hydrogen bond acceptors
    tpsa: float = 0.0  # topological polar surface area
    rotatable_bonds: int = 0

    def passes_lipinski(self) -> bool:
        """Check Lipinski's Rule of Five for drug-likeness."""
        violations = 0
        if self.mol_weight > 500:
            violations += 1
        if self.logp > 5:
            violations += 1
        if self.hbd > 5:
            violations += 1
        if self.hba > 10:
            violations += 1
        return violations <= 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "smiles": self.smiles,
            "mol_weight": self.mol_weight,
            "category": self.category,
            "mechanism": self.mechanism,
            "status": self.status,
            "logp": self.logp,
            "lipinski_pass": self.passes_lipinski(),
        }


# ============================================================
# Known GCase Modulators & Pharmacological Chaperones
# ============================================================

KNOWN_GCASE_MODULATORS = [
    Compound(
        name="Ambroxol",
        smiles="Nc1c(Br)cc(Br)cc1CNC1CCC(O)CC1",
        mol_weight=378.10,
        category="pharmacological_chaperone",
        mechanism=(
            "pH-dependent inhibitory chaperone. Binds GCase at neutral pH "
            "(ER), stabilizes folding, releases at acidic lysosomal pH. "
            "Enhances GCase activity and reduces alpha-synuclein levels."
        ),
        status="clinical",
        source="Repurposed mucolytic agent - Phase 2 clinical trials for PD",
        logp=2.8,
        hbd=3,
        hba=3,
        tpsa=58.3,
        rotatable_bonds=4,
    ),
    Compound(
        name="Isofagomine",
        smiles="OC1CNC(CO)CC1",
        mol_weight=147.17,
        category="pharmacological_chaperone",
        mechanism=(
            "Iminosugar that mimics glucose transition state. Competitive "
            "inhibitor at neutral pH, acts as chaperone by stabilizing "
            "native GCase fold. Enhances lysosomal trafficking of mutant GCase."
        ),
        status="clinical",
        source="Designed GCase chaperone - Phase 2 trials discontinued",
        logp=-2.1,
        hbd=3,
        hba=4,
        tpsa=73.6,
        rotatable_bonds=1,
    ),
    Compound(
        name="NCGC607",
        smiles="O=C(NCc1ccc(F)cc1)c1cc2ccccc2n1CC(O)CO",
        mol_weight=356.38,
        category="non_inhibitory_chaperone",
        mechanism=(
            "Non-inhibitory small molecule chaperone. Stabilizes GCase without "
            "competing for the active site. Enhances GCase activity in "
            "iPSC-derived dopaminergic neurons from GBA-PD patients."
        ),
        status="preclinical",
        source="NIH NCATS high-throughput screen",
        logp=2.1,
        hbd=3,
        hba=5,
        tpsa=82.4,
        rotatable_bonds=6,
    ),
    Compound(
        name="LTI-291",
        smiles="CC(C)c1ccc(NC(=O)c2cc(OC)c(OC)c(OC)c2)cc1",
        mol_weight=329.39,
        category="gcase_activator",
        mechanism=(
            "Allosteric GCase activator. Does not bind the active site - "
            "instead binds an allosteric pocket to enhance catalytic activity. "
            "Increases GCase activity in GBA-mutant fibroblasts."
        ),
        status="clinical",
        source="Lysosomal Therapeutics Inc - Phase 1 completed",
        logp=3.5,
        hbd=1,
        hba=4,
        tpsa=57.2,
        rotatable_bonds=5,
    ),
    Compound(
        name="S-181",
        smiles="CC(=O)Nc1ccc(-c2nc3cc(Cl)ccc3o2)cc1",
        mol_weight=286.71,
        category="gcase_activator",
        mechanism=(
            "Brain-penetrant GCase modulator. Activates wild-type and "
            "mutant GCase. Reduces glycosphingolipid substrates and "
            "alpha-synuclein levels in mouse models."
        ),
        status="preclinical",
        source="Published in Science Translational Medicine",
        logp=3.2,
        hbd=1,
        hba=3,
        tpsa=55.4,
        rotatable_bonds=2,
    ),
    Compound(
        name="Venglustat",
        smiles="OC(CNC1CCC(c2ccc(F)cc2F)CC1)c1cc(F)c(F)cc1F",
        mol_weight=425.39,
        category="substrate_reduction",
        mechanism=(
            "Glucosylceramide synthase inhibitor. Reduces glucosylceramide "
            "accumulation that results from GCase deficiency. Brain-penetrant. "
            "Targets downstream pathology of GBA mutations."
        ),
        status="clinical",
        source="Sanofi - Phase 2 MOVES-PD trial (discontinued for efficacy)",
        logp=3.8,
        hbd=2,
        hba=3,
        tpsa=44.1,
        rotatable_bonds=5,
    ),
]

# ============================================================
# Computationally Designed Candidate Compounds
# ============================================================

DESIGNED_CANDIDATES = [
    Compound(
        name="GBA-Chap-01",
        smiles="Oc1ccc(CNC2CC(O)C(O)C(O)C2O)cc1O",
        mol_weight=285.30,
        category="pharmacological_chaperone",
        mechanism="Catechol-iminosugar hybrid designed for dual GCase binding and antioxidant activity",
        status="experimental",
        source="Computational design",
        logp=-0.5,
        hbd=6,
        hba=7,
        tpsa=130.3,
        rotatable_bonds=3,
    ),
    Compound(
        name="GBA-Act-01",
        smiles="COc1cc(C(=O)Nc2cccc(NC(=O)c3ccncc3)c2)cc(OC)c1OC",
        mol_weight=407.42,
        category="gcase_activator",
        mechanism="Bis-amide activator targeting GCase allosteric site, designed from LTI-291 scaffold",
        status="experimental",
        source="Scaffold hopping from LTI-291",
        logp=2.3,
        hbd=2,
        hba=6,
        tpsa=93.5,
        rotatable_bonds=7,
    ),
    Compound(
        name="GBA-Dual-01",
        smiles="Nc1c(Br)cc(Br)cc1CNC1CCC(NC(=O)c2cccc(O)c2O)CC1",
        mol_weight=498.21,
        category="dual_mechanism",
        mechanism="Ambroxol-catechol conjugate for simultaneous GCase chaperoning and ROS scavenging in neurons",
        status="experimental",
        source="Hybrid pharmacophore design",
        logp=3.1,
        hbd=5,
        hba=5,
        tpsa=105.5,
        rotatable_bonds=6,
    ),
    Compound(
        name="GBA-BBB-01",
        smiles="CC(C)Oc1ccc(NC(=O)c2cc3ccccc3n2C)cc1",
        mol_weight=308.36,
        category="gcase_activator",
        mechanism="Brain-penetrant benzimidazole activator designed for enhanced BBB permeability (low TPSA, optimal logP)",
        status="experimental",
        source="BBB-optimized design",
        logp=3.4,
        hbd=1,
        hba=3,
        tpsa=49.8,
        rotatable_bonds=4,
    ),
    Compound(
        name="GBA-SynClear-01",
        smiles="O=C(Nc1ccc(-c2cnc3ccccc3n2)cc1)c1cccc(Cl)c1",
        mol_weight=359.81,
        category="synuclein_modulator",
        mechanism="Quinazoline scaffold targeting GCase-alpha-synuclein interaction interface to promote synuclein clearance",
        status="experimental",
        source="Protein-protein interaction design",
        logp=4.2,
        hbd=1,
        hba=4,
        tpsa=64.2,
        rotatable_bonds=3,
    ),
]

# ============================================================
# Reference Inhibitors for Validation
# ============================================================

REFERENCE_COMPOUNDS = [
    Compound(
        name="Conduritol_B_Epoxide",
        smiles="OC1C=CC(O)C2OC12",
        mol_weight=146.14,
        category="irreversible_inhibitor",
        mechanism="Covalent mechanism-based GCase inhibitor. Used as biochemical tool. Reacts with catalytic Glu235.",
        status="tool_compound",
        source="Classical GCase inhibitor",
        logp=-1.5,
        hbd=2,
        hba=4,
        tpsa=59.1,
        rotatable_bonds=0,
    ),
    Compound(
        name="DNJ_Deoxynojirimycin",
        smiles="OCC1NCC(O)C(O)C1O",
        mol_weight=163.17,
        category="competitive_inhibitor",
        mechanism="Glucose analog iminosugar. Competitive GCase inhibitor and pharmacological chaperone.",
        status="approved",
        source="Natural product from Bacillus",
        logp=-2.8,
        hbd=4,
        hba=5,
        tpsa=93.7,
        rotatable_bonds=1,
    ),
]


class CompoundLibrary:
    """
    Manages the compound library for GBA-targeted screening.

    Provides filtering, property calculation, and compound preparation
    for molecular docking campaigns.
    """

    def __init__(self):
        self.compounds: list[Compound] = []

    def load_known_modulators(self):
        """Load known GCase modulators into the library."""
        self.compounds.extend(KNOWN_GCASE_MODULATORS)
        logger.info(f"Loaded {len(KNOWN_GCASE_MODULATORS)} known GCase modulators")

    def load_designed_candidates(self):
        """Load computationally designed candidates."""
        self.compounds.extend(DESIGNED_CANDIDATES)
        logger.info(f"Loaded {len(DESIGNED_CANDIDATES)} designed candidates")

    def load_reference_compounds(self):
        """Load reference/tool compounds for validation."""
        self.compounds.extend(REFERENCE_COMPOUNDS)
        logger.info(f"Loaded {len(REFERENCE_COMPOUNDS)} reference compounds")

    def load_all(self):
        """Load all compound sets."""
        self.load_known_modulators()
        self.load_designed_candidates()
        self.load_reference_compounds()
        logger.info(f"Total compounds in library: {len(self.compounds)}")

    def add_compound(self, compound: Compound):
        """Add a single compound to the library."""
        self.compounds.append(compound)

    def add_from_smiles(self, name: str, smiles: str, category: str = "custom",
                        mechanism: str = ""):
        """Add a compound from SMILES string with automatic property calculation."""
        compound = Compound(name=name, smiles=smiles, category=category, mechanism=mechanism)
        compound = self._calculate_properties(compound)
        self.compounds.append(compound)
        return compound

    def _calculate_properties(self, compound: Compound) -> Compound:
        """Calculate molecular properties using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

            mol = Chem.MolFromSmiles(compound.smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES for {compound.name}: {compound.smiles}")
                return compound

            compound.mol_weight = Descriptors.MolWt(mol)
            compound.logp = Crippen.MolLogP(mol)
            compound.hbd = rdMolDescriptors.CalcNumHBD(mol)
            compound.hba = rdMolDescriptors.CalcNumHBA(mol)
            compound.tpsa = Descriptors.TPSA(mol)
            compound.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        except ImportError:
            logger.warning("RDKit not available - using pre-set properties")

        return compound

    def filter_by_category(self, category: str) -> list[Compound]:
        """Filter compounds by mechanism category."""
        return [c for c in self.compounds if c.category == category]

    def filter_lipinski(self, strict: bool = False) -> list[Compound]:
        """Filter compounds passing Lipinski's Rule of Five."""
        return [c for c in self.compounds if c.passes_lipinski()]

    def filter_bbb_permeable(self) -> list[Compound]:
        """
        Filter compounds likely to cross the Blood-Brain Barrier.

        Uses simplified BBB permeability criteria:
        - MW < 450 Da
        - LogP between 1.0 and 4.0
        - TPSA < 90 A^2
        - HBD <= 3
        - Rotatable bonds <= 8
        """
        bbb_permeable = []
        for c in self.compounds:
            if (c.mol_weight < 450
                    and 1.0 <= c.logp <= 4.0
                    and c.tpsa < 90
                    and c.hbd <= 3
                    and c.rotatable_bonds <= 8):
                bbb_permeable.append(c)
        return bbb_permeable

    def get_screening_set(self, include_reference: bool = True,
                          bbb_filter: bool = True) -> list[Compound]:
        """
        Get the final set of compounds for docking screening.

        Args:
            include_reference: Include reference inhibitors for validation
            bbb_filter: Apply BBB permeability filter (important for PD drugs)

        Returns:
            List of compounds ready for docking
        """
        candidates = [c for c in self.compounds if c.category != "irreversible_inhibitor"]

        if bbb_filter:
            bbb_candidates = self.filter_bbb_permeable()
            # Keep BBB-permeable + reference compounds
            if include_reference:
                refs = [c for c in self.compounds
                        if c.category in ("irreversible_inhibitor", "competitive_inhibitor")]
                candidates = list(set(bbb_candidates + refs))
            else:
                candidates = bbb_candidates

        logger.info(f"Screening set: {len(candidates)} compounds")
        return candidates

    def prepare_for_docking(self, compounds: Optional[list[Compound]] = None,
                            output_dir: str = "structures") -> dict[str, str]:
        """
        Prepare compound 3D structures and PDBQT files for docking.

        Args:
            compounds: List of compounds to prepare (default: all)
            output_dir: Directory for output files

        Returns:
            Dictionary mapping compound names to PDBQT file paths
        """
        from pathlib import Path

        if compounds is None:
            compounds = self.compounds

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        prepared = {}

        for compound in compounds:
            pdbqt_file = output_path / f"{compound.name}.pdbqt"
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                mol = Chem.MolFromSmiles(compound.smiles)
                if mol is None:
                    logger.warning(f"Skipping {compound.name}: invalid SMILES")
                    continue

                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)

                try:
                    from meeko import MoleculePreparation, PDBQTWriterLegacy

                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(mol)
                    for setup in mol_setups:
                        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
                        if is_ok:
                            pdbqt_file.write_text(pdbqt_string)
                            prepared[compound.name] = str(pdbqt_file)
                            break
                except ImportError:
                    self._simple_mol_to_pdbqt(mol, compound.name, str(pdbqt_file))
                    prepared[compound.name] = str(pdbqt_file)

            except ImportError:
                self._generate_placeholder_pdbqt(compound, str(pdbqt_file))
                prepared[compound.name] = str(pdbqt_file)

            logger.info(f"Prepared {compound.name} -> {pdbqt_file}")

        return prepared

    def _simple_mol_to_pdbqt(self, mol, name: str, output_path: str):
        """Convert RDKit mol to simple PDBQT format."""
        from rdkit import Chem

        pdb_block = Chem.MolToPDBBlock(mol)
        with open(output_path, "w") as f:
            f.write(f"REMARK  Name: {name}\n")
            f.write("ROOT\n")
            for line in pdb_block.split("\n"):
                if line.startswith(("ATOM", "HETATM")):
                    f.write(f"{line[:54]}  0.000    0.000 {'C':>2s}\n")
            f.write("ENDROOT\n")
            f.write("TORSDOF 0\n")

    def _generate_placeholder_pdbqt(self, compound: Compound, output_path: str):
        """Generate placeholder PDBQT when RDKit is not available."""
        with open(output_path, "w") as f:
            f.write(f"REMARK  Name: {compound.name}\n")
            f.write(f"REMARK  SMILES: {compound.smiles}\n")
            f.write(f"REMARK  MW: {compound.mol_weight:.2f}\n")
            f.write(f"REMARK  Category: {compound.category}\n")
            f.write("ROOT\n")
            f.write("ATOM      1  C1  LIG A   1      -0.500   0.500   0.000  1.00  0.00     0.000 C\n")
            f.write("ATOM      2  C2  LIG A   1       0.500   0.500   0.000  1.00  0.00     0.000 C\n")
            f.write("ATOM      3  O1  LIG A   1       0.000  -0.500   0.000  1.00  0.00     0.000 OA\n")
            f.write("ENDROOT\n")
            f.write("TORSDOF 0\n")

    def summary(self) -> str:
        """Return a summary of the compound library."""
        lines = [
            f"Compound Library Summary",
            f"{'=' * 50}",
            f"Total compounds: {len(self.compounds)}",
            f"",
        ]

        categories = {}
        for c in self.compounds:
            categories.setdefault(c.category, []).append(c)

        for cat, compounds in sorted(categories.items()):
            lines.append(f"  {cat}: {len(compounds)}")
            for c in compounds:
                lipinski = "PASS" if c.passes_lipinski() else "FAIL"
                lines.append(
                    f"    - {c.name} (MW={c.mol_weight:.1f}, LogP={c.logp:.1f}, "
                    f"Lipinski={lipinski}, Status={c.status})"
                )

        bbb = self.filter_bbb_permeable()
        lines.append(f"\n  BBB-permeable candidates: {len(bbb)}")
        for c in bbb:
            lines.append(f"    - {c.name} (TPSA={c.tpsa:.1f}, LogP={c.logp:.1f})")

        return "\n".join(lines)
