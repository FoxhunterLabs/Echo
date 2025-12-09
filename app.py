#!/usr/bin/env python3
"""
Echo: Serious-Safe, Single-File Recursive Decision-Support Engine

- Deterministic, local, human-gated.
- Hybrid runtime: cycle-based and event-driven modes.
- Decision-support ONLY. Do NOT connect directly to actuators or production control systems.

High-level modules (all in one file for your sanity):
- InputLayer              : explicit source intake
- NoveltyTriager          : benign/structural novelty tagging
- NormalizationMatrix     : numeric normalization + simple summaries
- TemporalAlignmentEngine : temporal drift vs historical EchoVectors
- MultiPathReasoner       : baseline / conservative / optimistic / historical risk lanes
- ShadowSimulator         : passive "next-cycle" risk heuristic
- PolicyLattice           : context-aware thresholds (day/night, weather, experience)
- JustificationSynthesizer: structured, deterministic explanations
- ProposalBuilder         : maps risk → band → recommendation
- HumanGate               : non-removable human approval gate
- ControlMatrix           : hard off-switch + audit + metadata persistence
- EchoVectorMemory        : compressed environment memory
- MetadataReservoir       : stores overrides & decisions
- EchoEngine              : orchestrator with hybrid runtime
"""

from __future__ import annotations

import csv
import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("echo")

Record = Dict[str, Any]
FeatureVector = Dict[str, float]


# -----------------------------------------------------------------------------
# Config & Core Data Structures
# -----------------------------------------------------------------------------

@dataclass
class SourceConfig:
    name: str
    type: str            # 'csv' | 'json'
    path: Optional[str]  # local path only
    trust: float = 0.8
    enabled: bool = True
    max_rows: int = 5000
    allowed_fields: Optional[List[str]] = None


@dataclass
class GovernanceConfig:
    require_human_approval: bool = True
    off_switch_default: bool = False  # engine starts enabled
    base_medium_threshold: float = 0.30
    base_high_threshold: float = 0.70


@dataclass
class PolicyContext:
    """Context variables that adjust risk thresholds."""
    domain: str = "safety"               # 'safety', 'ops', 'infra', etc.
    shift: str = "day"                   # 'day' | 'night'
    weather: str = "normal"             # 'normal' | 'adverse'
    operator_experience: str = "standard"  # 'new' | 'standard' | 'expert'


@dataclass
class EchoConfig:
    sources: List[SourceConfig]
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    policy_context: PolicyContext = field(default_factory=PolicyContext)
    metadata_dir: str = "./echo_metadata"
    run_id: Optional[str] = None


@dataclass
class Proposal:
    id: str
    description: str
    risk_score: float
    risk_band: str
    recommended_action: str
    justification: str


@dataclass
class DecisionOutcome:
    proposal_id: str
    approved: bool
    approved_by: Optional[str]
    timestamp: str
    notes: Optional[str] = None


@dataclass
class EchoMetadata:
    """
    Persisted across runs. Never used to control real systems, only to inform
    future risk assessments and drift analyses.
    """
    cycle_index: int = 0
    echo_vectors: List[Dict[str, Any]] = field(default_factory=list)
    drift_history: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    overrides: List[DecisionOutcome] = field(default_factory=list)
    weight_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EchoContext:
    config: EchoConfig
    metadata: EchoMetadata
    cycle_index: int
    mode: str  # 'cycle' | 'event'

    raw_records: List[Record] = field(default_factory=list)
    known_records: List[Record] = field(default_factory=list)
    novel_records: List[Record] = field(default_factory=list)

    features: List[FeatureVector] = field(default_factory=list)
    risk_summaries: List[Dict[str, Any]] = field(default_factory=list)

    drift_report: Optional[Dict[str, Any]] = None
    path_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    shadow_forecast: Optional[Dict[str, Any]] = None

    proposals: List[Proposal] = field(default_factory=list)
    decisions: List[DecisionOutcome] = field(default_factory=list)

    off_switch: bool = False


# -----------------------------------------------------------------------------
# Utility Helpers
# -----------------------------------------------------------------------------

def sha256_of_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# -----------------------------------------------------------------------------
# Input Layer (explicit sources only)
# -----------------------------------------------------------------------------

class InputLayer:
    """
    Intake is strictly limited to explicitly-configured sources.
    No discovery. No crawling. No network scanning.
    """

    def __init__(self, cfg: EchoConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("echo.InputLayer")

    def collect(self, mode: str) -> List[Record]:
        records: List[Record] = []
        for src in self.cfg.sources:
            if not src.enabled:
                continue
            if src.type == "csv" and src.path:
                records.extend(self._load_csv(src))
            elif src.type == "json" and src.path:
                records.extend(self._load_json(src))
            else:
                self.logger.warning("Unsupported or misconfigured source: %s", src)
        return records

    def _load_csv(self, src: SourceConfig) -> List[Record]:
        path = Path(src.path)
        if not path.exists():
            self.logger.warning("CSV not found for source %s: %s", src.name, src.path)
            return []
        out: List[Record] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= src.max_rows:
                    break
                out.append(self._filter_fields(row, src))
        self.logger.info("Loaded %d rows from %s", len(out), src.name)
        return out

    def _load_json(self, src: SourceConfig) -> List[Record]:
        path = Path(src.path)
        if not path.exists():
            self.logger.warning("JSON not found for source %s: %s", src.name, src.path)
            return []
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            self.logger.warning("JSON for %s was not list/dict; ignoring", src.name)
            return []
        out: List[Record] = []
        for i, item in enumerate(data):
            if i >= src.max_rows:
                break
            if isinstance(item, dict):
                out.append(self._filter_fields(item, src))
        self.logger.info("Loaded %d items from %s", len(out), src.name)
        return out

    def _filter_fields(self, row: Record, src: SourceConfig) -> Record:
        if src.allowed_fields is None:
            return dict(row)
        return {k: v for k, v in row.items() if k in src.allowed_fields}


# -----------------------------------------------------------------------------
# Novelty Triager (benign vs structural)
# -----------------------------------------------------------------------------

class NoveltyTriager:
    """
    Classifies novelty for logging/awareness. Does NOT auto-block or auto-route.
    """

    def __init__(self, metadata: EchoMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger("echo.NoveltyTriager")

    def process(self, records: List[Record]) -> Tuple[List[Record], List[Record]]:
        known: List[Record] = []
        novel: List[Record] = []
        seen = {a.get("hash") for a in self.metadata.anomalies if "hash" in a}

        for rec in records:
            h = sha256_of_obj(rec)
            if h in seen:
                known.append(rec)
                continue

            novelty_type = "benign"
            if self.metadata.anomalies:
                avg_keys = sum(len(a.get("record_keys", [])) for a in self.metadata.anomalies) / max(
                    len(self.metadata.anomalies), 1
                )
                if abs(len(rec.keys()) - avg_keys) > max(1.0, avg_keys * 0.5):
                    novelty_type = "structural"

            entry = {
                "hash": h,
                "first_seen": now_iso(),
                "record_keys": list(rec.keys()),
                "novelty_type": novelty_type,
            }
            self.metadata.anomalies.append(entry)
            novel.append(rec)

        if novel:
            self.logger.info("Novel records detected: %d", len(novel))

        # For now, we treat all data as usable; novelty is contextual only.
        return records, novel


# -----------------------------------------------------------------------------
# Normalization + Risk Summaries
# -----------------------------------------------------------------------------

class NormalizationMatrix:
    """
    Naive numeric normalization + simple summary stats per record.
    This is intentionally conservative: no ML, no black boxes.
    """

    def __init__(self, cfg: EchoConfig, metadata: EchoMetadata):
        self.cfg = cfg
        self.metadata = metadata
        self.logger = logging.getLogger("echo.NormalizationMatrix")

    def process(self, records: List[Record]) -> Tuple[List[FeatureVector], List[Dict[str, Any]]]:
        features: List[FeatureVector] = []
        summaries: List[Dict[str, Any]] = []

        numeric_fields = self._infer_numeric_fields(records)
        mins, maxs = self._compute_min_max(records, numeric_fields)

        for rec in records:
            fv: FeatureVector = {}
            for f in numeric_fields:
                v = rec.get(f)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                lo, hi = mins[f], maxs[f]
                if hi == lo:
                    fv[f] = 0.0
                else:
                    fv[f] = (v - lo) / (hi - lo)
            if fv:
                features.append(fv)
                summaries.append(
                    {
                        "mean": sum(fv.values()) / len(fv),
                        "max": max(fv.values()),
                        "min": min(fv.values()),
                        "field_count": len(fv),
                    }
                )

        weight_manifest = {
            "timestamp": now_iso(),
            "source_trust": {s.name: s.trust for s in self.cfg.sources},
        }
        self.metadata.weight_history.append(weight_manifest)
        self.logger.info("Generated %d feature vectors", len(features))
        return features, summaries

    @staticmethod
    def _infer_numeric_fields(records: List[Record]) -> List[str]:
        fields: List[str] = []
        for rec in records:
            for k, v in rec.items():
                try:
                    float(v)
                    if k not in fields:
                        fields.append(k)
                except Exception:
                    continue
        return fields

    @staticmethod
    def _compute_min_max(records: List[Record], fields: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        mins = {f: float("inf") for f in fields}
        maxs = {f: float("-inf") for f in fields}
        for rec in records:
            for f in fields:
                v = rec.get(f)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                mins[f] = min(mins[f], v)
                maxs[f] = max(maxs[f], v)
        for f in fields:
            if mins[f] == float("inf"):
                mins[f] = 0.0
            if maxs[f] == float("-inf"):
                maxs[f] = 1.0
        return mins, maxs


# -----------------------------------------------------------------------------
# Temporal Alignment Engine (drift)
# -----------------------------------------------------------------------------

class TemporalAlignmentEngine:
    """
    Compares current cycle's mean risk to historical mean, computes simple drift.
    """

    def __init__(self, metadata: EchoMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger("echo.TAE")

    def analyze(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {"status": "no_data", "drift_score": 0.0, "has_history": bool(self.metadata.echo_vectors)}

        current_mean = sum(s["mean"] for s in summaries) / len(summaries)

        if not self.metadata.echo_vectors:
            drift = 0.0
        else:
            past_means = [ev.get("mean_risk", 0.0) for ev in self.metadata.echo_vectors]
            hist_mean = sum(past_means) / len(past_means)
            drift = abs(current_mean - hist_mean)

        report = {
            "status": "ok",
            "current_mean_risk": current_mean,
            "drift_score": drift,
            "has_history": bool(self.metadata.echo_vectors),
        }
        self.metadata.drift_history.append(
            {"timestamp": now_iso(), "current_mean_risk": current_mean, "drift_score": drift}
        )
        self.logger.info("Temporal drift score: %.3f", drift)
        return report


# -----------------------------------------------------------------------------
# Multi-Path Reasoner
# -----------------------------------------------------------------------------

class MultiPathReasoner:
    """
    Computes multiple risk perspectives:
    - baseline      : raw mean
    - conservative  : slightly inflated risk
    - optimistic    : slightly deflated risk
    - historical    : blended with last mean
    """

    def __init__(self, metadata: EchoMetadata, gov: GovernanceConfig):
        self.metadata = metadata
        self.gov = gov
        self.logger = logging.getLogger("echo.MultiPathReasoner")

    def reason(self, summaries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        paths: Dict[str, List[Dict[str, Any]]] = {
            "baseline": [],
            "conservative": [],
            "optimistic": [],
            "historical": [],
        }
        if not summaries:
            return paths

        last_ev = self.metadata.echo_vectors[-1] if self.metadata.echo_vectors else None
        last_mean = last_ev.get("mean_risk", 0.0) if last_ev else 0.0

        for i, s in enumerate(summaries):
            base = s["mean"]
            paths["baseline"].append({"id": f"b_{i}", "risk": base})

            cons = min(1.0, base + 0.15)
            paths["conservative"].append({"id": f"c_{i}", "risk": cons})

            opt = max(0.0, base - 0.15)
            paths["optimistic"].append({"id": f"o_{i}", "risk": opt})

            hist_blend = 0.5 * base + 0.5 * last_mean
            paths["historical"].append({"id": f"h_{i}", "risk": hist_blend})

        self.logger.info("Multi-path reasoning computed for %d records", len(summaries))
        return paths


# -----------------------------------------------------------------------------
# Shadow Simulator
# -----------------------------------------------------------------------------

class ShadowSimulator:
    """
    Passive "next-step" heuristic. Does not act, only estimates trend.
    """

    def __init__(self, metadata: EchoMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger("echo.ShadowSimulator")

    def simulate_next(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {"status": "no_data"}

        current_mean = sum(s["mean"] for s in summaries) / len(summaries)
        drift = self.metadata.drift_history[-1]["drift_score"] if self.metadata.drift_history else 0.0
        forecast = current_mean + (0.2 * drift if drift > 0 else 0.0)
        forecast = max(0.0, min(1.0, forecast))

        result = {
            "status": "ok",
            "current_mean_risk": current_mean,
            "forecast_mean_risk": forecast,
            "used_drift": drift,
        }
        self.logger.info(
            "Shadow simulation: current=%.3f, forecast=%.3f, drift=%.3f",
            current_mean,
            forecast,
            drift,
        )
        return result


# -----------------------------------------------------------------------------
# Policy Lattice
# -----------------------------------------------------------------------------

class PolicyLattice:
    """
    Adjusts risk thresholds based on context (shift, weather, experience).
    """

    def __init__(self, gov: GovernanceConfig, ctx: PolicyContext):
        self.gov = gov
        self.ctx = ctx
        self.logger = logging.getLogger("echo.PolicyLattice")

    def thresholds(self) -> Dict[str, float]:
        med = self.gov.base_medium_threshold
        high = self.gov.base_high_threshold

        if self.ctx.shift == "night":
            med -= 0.05
            high -= 0.05
        if self.ctx.weather == "adverse":
            med -= 0.05
            high -= 0.10
        if self.ctx.operator_experience == "new":
            med -= 0.05
            high -= 0.05
        if self.ctx.operator_experience == "expert":
            med += 0.05
            high += 0.05

        med = max(0.0, min(1.0, med))
        high = max(0.0, min(1.0, high))
        self.logger.info("Policy thresholds => medium=%.2f, high=%.2f", med, high)
        return {"medium": med, "high": high}


# -----------------------------------------------------------------------------
# Justification Synthesizer
# -----------------------------------------------------------------------------

class JustificationSynthesizer:
    """
    Deterministic justification builder, no LLMs or stochastic text.
    """

    def __init__(self):
        self.logger = logging.getLogger("echo.JustificationSynthesizer")

    def build(
        self,
        risk: float,
        band: str,
        drift_report: Dict[str, Any],
        lane: str,
    ) -> str:
        parts = [
            f"Risk {risk:.2f} classified as {band.upper()} via {lane} lane.",
        ]
        if drift_report.get("status") == "ok":
            parts.append(
                f"Drift={drift_report.get('drift_score', 0.0):.2f}, "
                f"current_mean={drift_report.get('current_mean_risk', 0.0):.2f}."
            )
        else:
            parts.append("No historical baseline available; classification based on current cycle only.")
        return " ".join(parts)


# -----------------------------------------------------------------------------
# Proposal Builder
# -----------------------------------------------------------------------------

class ProposalBuilder:
    """
    Maps risk scores → bands → recommended actions.
    Uses baseline lane and policy thresholds.
    """

    def __init__(
        self,
        gov: GovernanceConfig,
        lattice: PolicyLattice,
        justifier: JustificationSynthesizer,
    ):
        self.gov = gov
        self.lattice = lattice
        self.justifier = justifier
        self.logger = logging.getLogger("echo.ProposalBuilder")

    def build(self, ctx: EchoContext) -> List[Proposal]:
        thresholds = self.lattice.thresholds()
        med_th, high_th = thresholds["medium"], thresholds["high"]
        baseline = ctx.path_results.get("baseline", [])

        proposals: List[Proposal] = []
        for item in baseline:
            risk = item["risk"]
            if risk >= high_th:
                band = "high"
                action = "STOP_AND_ESCALATE"
            elif risk >= med_th:
                band = "medium"
                action = "MITIGATE_AND_MONITOR"
            else:
                band = "low"
                action = "CONTINUE_WITH_CAUTION"

            justification = self.justifier.build(
                risk=risk,
                band=band,
                drift_report=ctx.drift_report or {"status": "no_data"},
                lane="baseline",
            )
            proposals.append(
                Proposal(
                    id=f"prop_{item['id']}",
                    description=f"Detected {band.upper()} risk scenario.",
                    risk_score=risk,
                    risk_band=band,
                    recommended_action=action,
                    justification=justification,
                )
            )

        self.logger.info("Built %d proposals from baseline path", len(proposals))
        return proposals


# -----------------------------------------------------------------------------
# Human Gate (non-removable)
# -----------------------------------------------------------------------------

class HumanGate:
    """
    All proposals MUST pass through this gate.
    No auto-approval. No bypass.
    """

    def __init__(self, gov: GovernanceConfig):
        self.gov = gov
        self.logger = logging.getLogger("echo.HumanGate")

    def review(self, proposals: List[Proposal]) -> List[DecisionOutcome]:
        outcomes: List[DecisionOutcome] = []
        if not proposals:
            print("\n[Echo] No proposals generated this run.")
            return outcomes

        print("\n========== ECHO PROPOSALS ==========")
        for prop in proposals:
            print(f"\nID: {prop.id}")
            print(f"  Risk Band        : {prop.risk_band.upper()}")
            print(f"  Risk Score       : {prop.risk_score:.2f}")
            print(f"  Recommended      : {prop.recommended_action}")
            print(f"  Description      : {prop.description}")
            print(f"  Justification    : {prop.justification}")

            while True:
                resp = input("Approve this proposal? [y/n]: ").strip().lower()
                if resp in ("y", "n"):
                    break
                print("Please enter 'y' or 'n'.")

            approved = (resp == "y")
            initials = input("Your initials (for audit): ").strip() or None
            notes = input("Optional notes: ").strip() or None

            outcomes.append(
                DecisionOutcome(
                    proposal_id=prop.id,
                    approved=approved,
                    approved_by=initials,
                    timestamp=now_iso(),
                    notes=notes,
                )
            )

        return outcomes


# -----------------------------------------------------------------------------
# Control Matrix & Echo Memory
# -----------------------------------------------------------------------------

class ControlMatrix:
    """
    Governance + hard off-switch + audit persistence.
    """

    def __init__(self, cfg: EchoConfig, metadata: EchoMetadata):
        self.cfg = cfg
        self.metadata = metadata
        self.metadata_dir = ensure_dir(cfg.metadata_dir)
        self.off_switch = cfg.governance.off_switch_default
        self.logger = logging.getLogger("echo.ControlMatrix")

    def maybe_toggle_off(self) -> None:
        resp = input("\nToggle HARD OFF-SWITCH for future runs? [y/N]: ").strip().lower()
        if resp == "y":
            self.off_switch = True
            self.logger.warning("HARD OFF-SWITCH ENGAGED.")
        else:
            self.logger.info("Off-switch remains: %s", self.off_switch)

    def persist_cycle_audit(self, ctx: EchoContext) -> None:
        audit = {
            "cycle_index": ctx.cycle_index,
            "mode": ctx.mode,
            "timestamp": now_iso(),
            "raw_record_count": len(ctx.raw_records),
            "proposal_count": len(ctx.proposals),
            "decision_count": len(ctx.decisions),
            "off_switch": ctx.off_switch,
            "metadata_hash": sha256_of_obj(self.metadata.to_dict()),
        }
        path = self.metadata_dir / f"audit_cycle_{ctx.cycle_index}_{ctx.mode}.json"
        with path.open("w") as f:
            json.dump(audit, f, indent=2)
        self.logger.info("Persisted audit record: %s", path)

    def persist_metadata(self) -> None:
        path = self.metadata_dir / "metadata.json"
        with path.open("w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        self.logger.info("Persisted metadata to %s", path)


class EchoVectorMemory:
    """
    Stores compressed environment descriptors (EchoVectors).
    """

    def __init__(self, metadata: EchoMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger("echo.EchoVectorMemory")

    def update(self, ctx: EchoContext) -> None:
        if not ctx.risk_summaries:
            return
        mean_risk = sum(s["mean"] for s in ctx.risk_summaries) / len(ctx.risk_summaries)
        ev = {
            "cycle_index": ctx.cycle_index,
            "mode": ctx.mode,
            "timestamp": now_iso(),
            "mean_risk": mean_risk,
            "drift_score": (ctx.drift_report or {}).get("drift_score", 0.0),
            "shadow_forecast": ctx.shadow_forecast,
        }
        self.metadata.echo_vectors.append(ev)
        self.logger.info("EchoVector updated for cycle %d", ctx.cycle_index)


class MetadataReservoir:
    """
    Collects decisions/overrides for later analysis.
    """

    def __init__(self, metadata: EchoMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger("echo.MetadataReservoir")

    def update(self, ctx: EchoContext) -> None:
        self.metadata.overrides.extend(ctx.decisions)
        self.logger.info("Metadata updated with %d decisions", len(ctx.decisions))


# -----------------------------------------------------------------------------
# Echo Engine (Hybrid Runtime)
# -----------------------------------------------------------------------------

class EchoEngine:
    """
    Orchestrates a full Echo run.
    Hybrid runtime: cycle-based and event-driven modes.
    """

    def __init__(self, cfg: EchoConfig):
        self.cfg = cfg
        self.metadata_dir = ensure_dir(cfg.metadata_dir)
        self.metadata = self._load_or_init_metadata()
        self.logger = logging.getLogger("echo.EchoEngine")

        # Subsystems
        self.input_layer = InputLayer(cfg)
        self.novelty = NoveltyTriager(self.metadata)
        self.norm = NormalizationMatrix(cfg, self.metadata)
        self.tae = TemporalAlignmentEngine(self.metadata)
        self.mpr = MultiPathReasoner(self.metadata, cfg.governance)
        self.shadow = ShadowSimulator(self.metadata)
        self.lattice = PolicyLattice(cfg.governance, cfg.policy_context)
        self.justifier = JustificationSynthesizer()
        self.builder = ProposalBuilder(cfg.governance, self.lattice, self.justifier)
        self.human_gate = HumanGate(cfg.governance)
        self.control = ControlMatrix(cfg, self.metadata)
        self.echo_mem = EchoVectorMemory(self.metadata)
        self.meta_res = MetadataReservoir(self.metadata)

    def _load_or_init_metadata(self) -> EchoMetadata:
        path = self.metadata_dir / "metadata.json"
        if path.exists():
            with path.open("r") as f:
                raw = json.load(f)
            logger.info("Loaded existing Echo metadata from %s", path)
            return EchoMetadata(
                cycle_index=raw.get("cycle_index", 0),
                echo_vectors=raw.get("echo_vectors", []),
                drift_history=raw.get("drift_history", []),
                anomalies=raw.get("anomalies", []),
                overrides=raw.get("overrides", []),
                weight_history=raw.get("weight_history", []),
            )
        logger.info("Initializing new Echo metadata.")
        return EchoMetadata()

    def run(self, mode: str) -> None:
        """
        mode: 'cycle' (scheduled tick) or 'event' (external trigger).
        """
        assert mode in ("cycle", "event")
        if self.control.off_switch:
            self.logger.warning("OFF-SWITCH engaged. Skipping run (%s).", mode)
            return

        cycle_index = self.metadata.cycle_index + 1
        ctx = EchoContext(
            config=self.cfg,
            metadata=self.metadata,
            cycle_index=cycle_index,
            mode=mode,
        )
        self.logger.info("=== Echo %s run %d ===", mode.upper(), cycle_index)

        # 1. Intake
        ctx.raw_records = self.input_layer.collect(mode=mode)

        # 2. Novelty tagging
        ctx.known_records, ctx.novel_records = self.novelty.process(ctx.raw_records)

        # 3. Normalization
        ctx.features, ctx.risk_summaries = self.norm.process(ctx.known_records)

        # 4. Temporal alignment
        ctx.drift_report = self.tae.analyze(ctx.risk_summaries)

        # 5. Multi-path reasoning
        ctx.path_results = self.mpr.reason(ctx.risk_summaries)

        # 6. Shadow simulation
        ctx.shadow_forecast = self.shadow.simulate_next(ctx.risk_summaries)

        # 7. Proposal construction
        ctx.proposals = self.builder.build(ctx)

        # 8. Human gate (non-removable)
        ctx.decisions = self.human_gate.review(ctx.proposals)

        # 9. Metadata & memory updates
        self.meta_res.update(ctx)
        self.echo_mem.update(ctx)

        # 10. Audit & persistence
        ctx.off_switch = self.control.off_switch
        self.control.persist_cycle_audit(ctx)
        self.control.persist_metadata()
        self.control.maybe_toggle_off()

        # 11. Advance cycle index
        self.metadata.cycle_index = cycle_index
        self.logger.info("=== Echo run %d (%s) complete ===", cycle_index, mode)


# -----------------------------------------------------------------------------
# Default Config & CLI
# -----------------------------------------------------------------------------

def build_default_config() -> EchoConfig:
    """
    Safe default:
    - One local CSV at ./data/input.csv if it exists.
    - No network IO.
    """
    sources = [
        SourceConfig(
            name="local_csv",
            type="csv",
            path="./data/input.csv",
            trust=0.9,
            allowed_fields=None,
            enabled=True,
            max_rows=1000,
        )
    ]
    gov = GovernanceConfig()
    policy_ctx = PolicyContext(
        domain="safety",
        shift="day",
        weather="normal",
        operator_experience="standard",
    )
    return EchoConfig(
        sources=sources,
        governance=gov,
        policy_context=policy_ctx,
        metadata_dir="./echo_metadata",
    )


def main():
    cfg = build_default_config()
    engine = EchoEngine(cfg)

    print("Echo – Serious-Safe Recursive Decision-Support Engine (Single-File)")
    print("NOTE: This is decision-support only. Do NOT connect directly to actuators.\n")

    try:
        while True:
            print("\nSelect mode:")
            print("  [1] Scheduled cycle run")
            print("  [2] Event-driven run")
            print("  [3] Quit")
            choice = input("Enter choice [1/2/3]: ").strip()

            if choice == "1":
                engine.run(mode="cycle")
            elif choice == "2":
                engine.run(mode="event")
            elif choice == "3":
                break
            else:
                print("Invalid choice.")
    except KeyboardInterrupt:
        print("\nExiting Echo.")
    finally:
        engine.control.persist_metadata()


if __name__ == "__main__":
    main()
