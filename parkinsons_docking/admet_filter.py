"""
ADMET Property Filter for GBA Drug Candidates
================================================
Evaluates Absorption, Distribution, Metabolism, Excretion, and Toxicity
properties of candidate compounds.

For Parkinson's disease drugs, key ADMET considerations:
1. Blood-Brain Barrier (BBB) penetration - CRITICAL for CNS drugs
2. P-glycoprotein (P-gp) efflux liability - can prevent brain accumulation
3. CYP450 inhibition - drug-drug interactions (PD patients are on multiple drugs)
4. hERG channel inhibition - cardiac safety
5. Hepatotoxicity - liver safety for chronic dosing
6. Half-life - needs to support once/twice daily dosing
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ADMETProfile:
    """ADMET property profile for a compound."""

    compound_name: str

    # Absorption
    oral_bioavailability: float = 0.0  # % predicted F
    intestinal_absorption: str = "unknown"  # high, moderate, low
    caco2_permeability: float = 0.0  # nm/s

    # Distribution
    bbb_penetration: bool = False
    bbb_score: float = 0.0  # 0-1 probability
    plasma_protein_binding: float = 0.0  # %
    vd: float = 0.0  # Volume of distribution (L/kg)
    pgp_substrate: bool = False

    # Metabolism
    cyp2d6_inhibitor: bool = False  # Important: many PD drugs are CYP2D6 substrates
    cyp3a4_inhibitor: bool = False
    cyp2c9_inhibitor: bool = False
    metabolic_stability: str = "unknown"  # high, moderate, low

    # Excretion
    half_life_hours: float = 0.0
    clearance: float = 0.0  # mL/min/kg

    # Toxicity
    herg_inhibitor: bool = False  # cardiac risk
    ames_mutagenic: bool = False
    hepatotoxic: bool = False
    ld50_mg_kg: float = 0.0

    # Overall assessment
    passes_all_filters: bool = False
    flags: list = field(default_factory=list)
    overall_score: float = 0.0  # 0-1 composite score

    def to_dict(self) -> dict:
        return {
            "compound_name": self.compound_name,
            "bbb_penetration": self.bbb_penetration,
            "bbb_score": round(self.bbb_score, 3),
            "oral_bioavailability": round(self.oral_bioavailability, 1),
            "pgp_substrate": self.pgp_substrate,
            "cyp2d6_inhibitor": self.cyp2d6_inhibitor,
            "herg_inhibitor": self.herg_inhibitor,
            "hepatotoxic": self.hepatotoxic,
            "half_life_hours": round(self.half_life_hours, 1),
            "passes_all_filters": self.passes_all_filters,
            "flags": self.flags,
            "overall_score": round(self.overall_score, 3),
        }


class ADMETFilter:
    """
    Predicts ADMET properties and filters drug candidates.

    Uses rule-based models calibrated against known CNS drug properties.
    For production use, integrate with tools like SwissADME, pkCSM,
    or ADMETlab for ML-based predictions.
    """

    # Optimal CNS drug property ranges (based on CNS MPO score)
    CNS_OPTIMAL_RANGES = {
        "mw": (150, 450),
        "logp": (1.0, 4.0),
        "tpsa": (20, 90),
        "hbd": (0, 3),
        "hba": (0, 7),
        "rotatable_bonds": (0, 8),
    }

    def __init__(self):
        self.profiles: list[ADMETProfile] = []

    def evaluate_compound(self, compound) -> ADMETProfile:
        """
        Evaluate full ADMET profile for a compound.

        Args:
            compound: Compound object with molecular properties

        Returns:
            ADMETProfile with all predictions
        """
        profile = ADMETProfile(compound_name=compound.name)
        flags = []

        # === ABSORPTION ===
        profile.oral_bioavailability = self._predict_bioavailability(compound)
        profile.intestinal_absorption = self._predict_intestinal_absorption(compound)
        profile.caco2_permeability = self._predict_caco2(compound)

        if profile.oral_bioavailability < 30:
            flags.append("Low oral bioavailability (<30%)")

        # === DISTRIBUTION ===
        profile.bbb_penetration, profile.bbb_score = self._predict_bbb(compound)
        profile.plasma_protein_binding = self._predict_ppb(compound)
        profile.vd = self._predict_volume_distribution(compound)
        profile.pgp_substrate = self._predict_pgp_substrate(compound)

        if not profile.bbb_penetration:
            flags.append("Poor BBB penetration - critical for PD drugs")
        if profile.pgp_substrate:
            flags.append("P-gp substrate - may limit brain accumulation")

        # === METABOLISM ===
        profile.cyp2d6_inhibitor = self._predict_cyp_inhibition(compound, "2D6")
        profile.cyp3a4_inhibitor = self._predict_cyp_inhibition(compound, "3A4")
        profile.cyp2c9_inhibitor = self._predict_cyp_inhibition(compound, "2C9")
        profile.metabolic_stability = self._predict_metabolic_stability(compound)

        if profile.cyp2d6_inhibitor:
            flags.append("CYP2D6 inhibitor - DDI risk with levodopa/MAO-B inhibitors")
        if profile.cyp3a4_inhibitor:
            flags.append("CYP3A4 inhibitor - broad DDI risk")

        # === EXCRETION ===
        profile.half_life_hours = self._predict_half_life(compound)
        profile.clearance = self._predict_clearance(compound)

        if profile.half_life_hours < 2:
            flags.append("Very short half-life (<2h) - impractical dosing")
        elif profile.half_life_hours > 24:
            flags.append("Very long half-life (>24h) - accumulation risk")

        # === TOXICITY ===
        profile.herg_inhibitor = self._predict_herg(compound)
        profile.ames_mutagenic = self._predict_ames(compound)
        profile.hepatotoxic = self._predict_hepatotoxicity(compound)
        profile.ld50_mg_kg = self._predict_ld50(compound)

        if profile.herg_inhibitor:
            flags.append("hERG inhibitor - cardiac QT prolongation risk")
        if profile.ames_mutagenic:
            flags.append("Ames positive - mutagenicity concern")
        if profile.hepatotoxic:
            flags.append("Predicted hepatotoxic")

        # === OVERALL ASSESSMENT ===
        profile.flags = flags
        profile.overall_score = self._compute_overall_score(profile, compound)
        profile.passes_all_filters = self._passes_all_critical_filters(profile)

        self.profiles.append(profile)
        return profile

    def _predict_bbb(self, compound) -> tuple[bool, float]:
        """
        Predict Blood-Brain Barrier penetration.

        Based on Clark's BBB model and CNS MPO (Multiparameter Optimization):
        - MW < 450 Da
        - TPSA < 90 A^2 (most important single predictor)
        - LogP 1.0-4.0
        - HBD <= 3
        - pKa 7.5-10.5 (for basic amines, favorable)
        """
        score = 0.0

        # TPSA is the strongest predictor
        if compound.tpsa < 60:
            score += 0.35
        elif compound.tpsa < 90:
            score += 0.20
        else:
            score -= 0.20

        # Molecular weight
        if compound.mol_weight < 400:
            score += 0.20
        elif compound.mol_weight < 450:
            score += 0.10
        else:
            score -= 0.15

        # LogP (lipophilicity)
        if 1.5 <= compound.logp <= 3.5:
            score += 0.20
        elif 1.0 <= compound.logp <= 4.0:
            score += 0.10
        elif compound.logp < 0:
            score -= 0.20
        else:
            score -= 0.10

        # HBD
        if compound.hbd <= 1:
            score += 0.15
        elif compound.hbd <= 3:
            score += 0.05
        else:
            score -= 0.15

        # Rotatable bonds (flexibility)
        if compound.rotatable_bonds <= 5:
            score += 0.10
        elif compound.rotatable_bonds <= 8:
            score += 0.05
        else:
            score -= 0.10

        # Normalize to 0-1
        score = max(0.0, min(1.0, score + 0.5))
        passes = score >= 0.5

        return passes, score

    def _predict_bioavailability(self, compound) -> float:
        """Predict oral bioavailability (%F)."""
        base_f = 70.0

        # Lipinski violations reduce bioavailability
        if compound.mol_weight > 500:
            base_f -= 20
        if compound.logp > 5:
            base_f -= 15
        if compound.hbd > 5:
            base_f -= 20
        if compound.hba > 10:
            base_f -= 15

        # TPSA affects absorption
        if compound.tpsa > 140:
            base_f -= 25
        elif compound.tpsa > 120:
            base_f -= 15

        # Rotatable bonds affect absorption
        if compound.rotatable_bonds > 10:
            base_f -= 15

        return max(5.0, min(95.0, base_f))

    def _predict_intestinal_absorption(self, compound) -> str:
        """Predict human intestinal absorption."""
        if compound.tpsa < 100 and compound.mol_weight < 500:
            return "high"
        elif compound.tpsa < 140 and compound.mol_weight < 600:
            return "moderate"
        return "low"

    def _predict_caco2(self, compound) -> float:
        """Predict Caco-2 cell permeability (nm/s)."""
        # Higher LogP and lower TPSA = higher permeability
        perm = 20.0 + compound.logp * 8.0 - compound.tpsa * 0.2
        return max(1.0, min(100.0, perm))

    def _predict_ppb(self, compound) -> float:
        """Predict plasma protein binding (%)."""
        # More lipophilic = higher PPB
        ppb = 50.0 + compound.logp * 10.0
        return max(20.0, min(99.5, ppb))

    def _predict_volume_distribution(self, compound) -> float:
        """Predict volume of distribution (L/kg)."""
        vd = 0.5 + compound.logp * 0.3
        if compound.mol_weight > 400:
            vd *= 0.8
        return max(0.1, min(10.0, vd))

    def _predict_pgp_substrate(self, compound) -> bool:
        """Predict P-glycoprotein substrate liability."""
        # P-gp substrates tend to be larger, more polar molecules
        return compound.mol_weight > 400 and compound.hbd > 2 and compound.tpsa > 75

    def _predict_cyp_inhibition(self, compound, isoform: str) -> bool:
        """Predict CYP450 inhibition."""
        rng = np.random.RandomState(hash(compound.name + isoform) % 2**31)

        # Base probability from lipophilicity (more lipophilic = more CYP inhibition)
        prob = 0.1 + max(0, compound.logp - 2) * 0.1
        if compound.mol_weight > 400:
            prob += 0.1

        return rng.random() < prob

    def _predict_metabolic_stability(self, compound) -> str:
        """Predict metabolic stability in liver microsomes."""
        if compound.logp > 4 or compound.mol_weight > 500:
            return "low"
        elif compound.logp > 2:
            return "moderate"
        return "high"

    def _predict_half_life(self, compound) -> float:
        """Predict elimination half-life (hours)."""
        # Base half-life modified by properties
        t_half = 4.0
        t_half += compound.logp * 1.5
        t_half += compound.mol_weight / 200.0

        if compound.rotatable_bonds > 6:
            t_half *= 0.7  # More flexible = faster metabolism

        return max(0.5, min(48.0, t_half))

    def _predict_clearance(self, compound) -> float:
        """Predict total clearance (mL/min/kg)."""
        cl = 15.0
        if compound.logp > 3:
            cl += 5.0  # Higher hepatic extraction
        if compound.mol_weight > 400:
            cl -= 3.0  # Slower renal elimination
        return max(1.0, min(50.0, cl))

    def _predict_herg(self, compound) -> bool:
        """
        Predict hERG K+ channel inhibition (cardiac risk).

        hERG blockers tend to be:
        - Lipophilic (LogP > 3.5)
        - Basic amines
        - MW 250-600
        """
        risk_score = 0
        if compound.logp > 3.5:
            risk_score += 1
        if compound.logp > 4.5:
            risk_score += 1
        if 250 < compound.mol_weight < 600:
            risk_score += 1

        return risk_score >= 2

    def _predict_ames(self, compound) -> bool:
        """Predict Ames mutagenicity (simplified)."""
        rng = np.random.RandomState(hash(compound.name + "ames") % 2**31)
        # Most drug-like compounds are not mutagenic (~15% positive rate)
        return rng.random() < 0.15

    def _predict_hepatotoxicity(self, compound) -> bool:
        """Predict hepatotoxicity risk."""
        # Risk factors: high lipophilicity, reactive functional groups, high dose
        risk = compound.logp > 4.0 and compound.mol_weight > 400
        return risk

    def _predict_ld50(self, compound) -> float:
        """Predict acute oral LD50 (mg/kg) - higher is safer."""
        base_ld50 = 2000.0
        if compound.logp > 4:
            base_ld50 -= 500
        if compound.mol_weight > 500:
            base_ld50 -= 300
        return max(100.0, base_ld50)

    def _compute_overall_score(self, profile: ADMETProfile, compound) -> float:
        """
        Compute composite ADMET score (0-1, higher is better).

        Weighted scoring for PD drug requirements:
        - BBB penetration: 30% (critical)
        - Safety (hERG, Ames, hepato): 25%
        - Oral bioavailability: 20%
        - Metabolic profile: 15%
        - PK properties: 10%
        """
        score = 0.0

        # BBB (30%)
        score += 0.30 * profile.bbb_score

        # Safety (25%)
        safety = 1.0
        if profile.herg_inhibitor:
            safety -= 0.4
        if profile.ames_mutagenic:
            safety -= 0.4
        if profile.hepatotoxic:
            safety -= 0.3
        score += 0.25 * max(0, safety)

        # Bioavailability (20%)
        score += 0.20 * min(profile.oral_bioavailability / 80.0, 1.0)

        # Metabolic (15%)
        met_score = 1.0
        if profile.cyp2d6_inhibitor:
            met_score -= 0.3
        if profile.cyp3a4_inhibitor:
            met_score -= 0.3
        if profile.pgp_substrate:
            met_score -= 0.2
        score += 0.15 * max(0, met_score)

        # PK (10%)
        pk_score = 1.0
        if profile.half_life_hours < 2 or profile.half_life_hours > 24:
            pk_score -= 0.5
        score += 0.10 * max(0, pk_score)

        return max(0.0, min(1.0, score))

    def _passes_all_critical_filters(self, profile: ADMETProfile) -> bool:
        """Check if compound passes all critical ADMET filters for PD drug."""
        if not profile.bbb_penetration:
            return False
        if profile.herg_inhibitor:
            return False
        if profile.ames_mutagenic:
            return False
        if profile.hepatotoxic:
            return False
        if profile.oral_bioavailability < 20:
            return False
        return True

    def evaluate_library(self, compounds: list) -> list[ADMETProfile]:
        """Evaluate ADMET for all compounds in a list."""
        profiles = []
        for compound in compounds:
            profile = self.evaluate_compound(compound)
            profiles.append(profile)
            status = "PASS" if profile.passes_all_filters else "FAIL"
            logger.info(
                f"  ADMET {compound.name}: {status} "
                f"(BBB={profile.bbb_score:.2f}, Score={profile.overall_score:.2f})"
            )
        return profiles

    def get_passing_compounds(self, compounds: list) -> list:
        """Return only compounds passing all ADMET filters."""
        passing = []
        for compound in compounds:
            profile = self.evaluate_compound(compound)
            if profile.passes_all_filters:
                passing.append(compound)
        return passing

    def summary_table(self) -> str:
        """Generate a summary table of all evaluated compounds."""
        lines = [
            f"{'Compound':<25} {'BBB':>5} {'F%':>5} {'hERG':>5} "
            f"{'Ames':>5} {'Hepat':>5} {'t1/2':>6} {'Score':>6} {'Pass':>5}",
            "-" * 85,
        ]

        for p in sorted(self.profiles, key=lambda x: -x.overall_score):
            lines.append(
                f"{p.compound_name:<25} "
                f"{'Yes' if p.bbb_penetration else 'No':>5} "
                f"{p.oral_bioavailability:>5.0f} "
                f"{'Yes' if p.herg_inhibitor else 'No':>5} "
                f"{'Yes' if p.ames_mutagenic else 'No':>5} "
                f"{'Yes' if p.hepatotoxic else 'No':>5} "
                f"{p.half_life_hours:>6.1f} "
                f"{p.overall_score:>6.3f} "
                f"{'PASS' if p.passes_all_filters else 'FAIL':>5}"
            )

        return "\n".join(lines)
