"""Generate the report's figures from the results folders (P4.3).

Regenerating from `results/` rather than hand-drawing means a figure can never drift
from the numbers it claims to show, and a re-run updates the paper by re-running this.

Every panel carries its uncertainty. On this benchmark that is not decoration: attack
success is a proportion over 100 targets, so each point is about +-0.09, and a reader
comparing two points without that will conclude things the data does not support.

    python -m report.make_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless box
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from eval.metrics import proportion_interval  # noqa: E402

RESULTS = Path("results")
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)

# A colourblind-safe set; the figures are also distinguishable by marker, because a
# printed or photocopied page loses colour before it loses shape.
COLOURS = {
    "no_memory": "#444444",
    "memory": "#d55e00",
    "gate_only": "#0072b2",
    "trust_only": "#009e73",
    "gated_memory": "#cc79a7",
    "undefended": "#d55e00",
    "source_trust": "#009e73",
}
MARKERS = {
    "no_memory": "s",
    "memory": "o",
    "gate_only": "^",
    "trust_only": "v",
    "gated_memory": "D",
    "undefended": "o",
    "source_trust": "v",
}
LABELS = {
    "no_memory": "no memory",
    "memory": "memory (ExpRAG)",
    "gate_only": "+ gating (C2)",
    "trust_only": "+ trust (C3)",
    "gated_memory": "+ both (C2+C3)",
    "undefended": "undefended",
    "source_trust": "+ trust (C3)",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name / "metrics.json").read_text(encoding="utf-8"))


def _series(
    cells: list[dict], config: str, xkey: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    rows = [c for c in cells if c["config"] == config]
    xs = [c[xkey] for c in rows]
    ys = [c["attack_success_rate"] for c in rows]
    lo, hi = [], []
    for c in rows:
        iv = proportion_interval(round(c["attack_success_rate"] * 100), 100, n_resamples=4000)
        lo.append(iv.point - iv.low)
        hi.append(iv.high - iv.point)
    return xs, ys, lo, hi


def fig_memory_poisoning() -> Path:
    """H1 and H2 in one panel: the dose-response curve and what each defence does to it."""
    d = _load("2026-10-28_poison-ablation-naive")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for config in ("memory", "gate_only", "trust_only", "gated_memory"):
        xs, ys, lo, hi = _series(d["cells"], config, "rate")
        if not xs:
            continue
        ax.errorbar(
            [x * 100 for x in xs],
            ys,
            yerr=[lo, hi],
            marker=MARKERS[config],
            color=COLOURS[config],
            label=LABELS[config],
            capsize=3,
            linewidth=1.6,
            markersize=6,
        )

    # no-memory has no store to poison, so it is a horizontal reference, not a curve.
    nm = next(c for c in d["cells"] if c["config"] == "no_memory")
    ax.axhline(nm["attack_success_rate"], color=COLOURS["no_memory"], linestyle=":", linewidth=1.5)
    ax.annotate(
        f"no memory ({nm['attack_success_rate']:.2f}) — nothing to poison",
        xy=(1.0, nm["attack_success_rate"]),
        xytext=(1.0, nm["attack_success_rate"] + 0.022),
        fontsize=8,
        color=COLOURS["no_memory"],
    )

    ax.set_xlabel("poisoned share of the experience store (%)")
    ax.set_ylabel("attack success rate")
    ax.set_title(
        "Memory poisoning: 100 targets, AVeriTeC dev\n"
        "naive adversary; bars are 95% bootstrap intervals",
        fontsize=10,
    )
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    path = OUT / "memory_poisoning.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_adversary_inversion() -> Path:
    """The C1 finding: which defence carries H2 swaps when the adversary adapts."""
    naive = _load("2026-10-28_poison-ablation-naive")
    adapt = _load("2026-10-28_poison-adaptive")

    def at(d: dict, config: str, rate: float = 0.2) -> float:
        return next(
            c["attack_success_rate"]
            for c in d["cells"]
            if c["config"] == config and abs(c["rate"] - rate) < 1e-9
        )

    configs = ("memory", "gate_only", "trust_only", "gated_memory")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    width = 0.36
    xs = range(len(configs))

    # The legend below is built by hand, so the adversary labels live there, not here.
    for off, (d, hatch) in enumerate(((naive, ""), (adapt, "///"))):
        vals = [at(d, c) for c in configs]
        errs = [
            [v - proportion_interval(round(v * 100), 100, n_resamples=4000).low for v in vals],
            [proportion_interval(round(v * 100), 100, n_resamples=4000).high - v for v in vals],
        ]
        ax.bar(
            [x + (off - 0.5) * width for x in xs],
            vals,
            width,
            yerr=errs,
            capsize=3,
            hatch=hatch,
            color=["#d55e00", "#0072b2", "#009e73", "#cc79a7"],
            edgecolor="white",
            alpha=0.95,
        )

    nm = next(c["attack_success_rate"] for c in naive["cells"] if c["config"] == "no_memory")
    ax.axhline(nm, color="#444444", linestyle=":", linewidth=1.5)
    ax.annotate(
        f"no memory ({nm:.2f})", xy=(len(configs) - 1.4, nm + 0.015), fontsize=8, color="#444444"
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=8)
    ax.set_ylabel("attack success rate at 20% poisoning")
    ax.set_title(
        "Trust defends against a stranger, not an impersonator\n"
        "the adaptive adversary cites the domains the history trusts most",
        fontsize=10,
    )
    ax.set_ylim(0, 0.65)
    # Colour already encodes the configuration, which is on the x-axis, so the legend
    # must be about the hatch alone. Coloured swatches would imply each adversary has a
    # colour, and they do not.
    ax.legend(
        handles=[
            Patch(facecolor="#bbbbbb", edgecolor="white", label="naive adversary"),
            Patch(facecolor="#bbbbbb", edgecolor="white", hatch="///", label="adaptive adversary"),
        ],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    ax.grid(alpha=0.25, axis="y", linewidth=0.5)
    fig.tight_layout()
    path = OUT / "adversary_inversion.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_source_poisoning() -> Path:
    """Source poisoning, and the reason gating cannot defend it."""
    d = _load("2026-10-29_source-poison-naive")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    for config in ("undefended", "source_trust"):
        xs, ys, lo, hi = _series(d["cells"], config, "pages_per_target")
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            marker=MARKERS[config],
            color=COLOURS[config],
            label=LABELS[config],
            capsize=3,
            linewidth=1.6,
            markersize=6,
        )
    ax.set_xlabel("fabricated pages per target")
    ax.set_ylabel("attack success rate")
    ax.set_title("Source poisoning", fontsize=10)
    ax.set_ylim(0, 0.82)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)

    # The evaluator's grade on fabricated vs real evidence. If these coincide, relevance
    # grading has no signal to act on -- which is the argument, measured.
    # Only the undefended cells: that is what the evaluator makes of the fabrications
    # before trust removes any of them. Including the defended cells draws two series as
    # one, and the lines cross meaninglessly.
    rows = [
        c
        for c in d["cells"]
        if c["config"] == "undefended" and c["grades"].get("mean_grade_fake") is not None
    ]
    pages = [c["pages_per_target"] for c in rows]
    fake = [c["grades"]["mean_grade_fake"] for c in rows]
    real = [c["grades"]["mean_grade_real"] for c in rows]
    bx.plot(pages, real, marker="s", color="#0072b2", label="real evidence", linewidth=1.6)
    bx.plot(pages, fake, marker="X", color="#d55e00", label="fabricated evidence", linewidth=1.6)
    bx.fill_between(pages, fake, real, color="#999999", alpha=0.18)
    for p, f, r in zip(pages, fake, real, strict=True):
        bx.annotate(
            f"gap {r - f:+.3f}",
            xy=(p, max(f, r) + 0.03),
            fontsize=7.5,
            ha="center",
            color="#555555",
        )
    bx.set_xlabel("fabricated pages per target")
    bx.set_ylabel("mean relevance grade")
    bx.set_title(
        "Why gating cannot help: the evaluator\nscores fabrications as relevant", fontsize=10
    )
    bx.set_ylim(0.3, 1.0)
    bx.set_xticks(pages)
    bx.legend(fontsize=8, loc="lower left", frameon=False)
    bx.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    path = OUT / "source_poisoning.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_g1_intervals() -> Path:
    """The Phase 1 table with its uncertainty, which is the honest version of it."""
    from eval.metrics import bootstrap_metric, pairs_from_confusion

    d = _load("2026-10-28_g1-rerun")
    systems = [s for s in ("no_retrieval", "plain_rag", "crag") if s in d]
    names = {"no_retrieval": "no retrieval", "plain_rag": "plain RAG", "crag": "CRAG"}

    points, los, his = [], [], []
    for s in systems:
        pairs = pairs_from_confusion(d[s]["confusion"])
        iv = bootstrap_metric([g for g, _ in pairs], [p for _, p in pairs], n_resamples=4000)
        points.append(iv.point)
        los.append(iv.point - iv.low)
        his.append(iv.high - iv.point)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ys = range(len(systems))
    ax.errorbar(
        points,
        list(ys),
        xerr=[los, his],
        fmt="o",
        color="#0072b2",
        capsize=4,
        markersize=7,
        linewidth=1.6,
    )
    maj = 0.189  # majority-class floor, recorded in PROGRESS
    ax.axvline(maj, color="#999999", linestyle=":", linewidth=1.4)
    ax.annotate("majority class", xy=(maj + 0.004, -0.42), fontsize=8, color="#777777")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([names[s] for s in systems])
    ax.set_xlabel("macro-F1 (95% bootstrap interval, n=500)")
    ax.set_title(
        "Phase 1 on AVeriTeC dev: the intervals overlap\nno pairwise difference excludes zero",
        fontsize=10,
    )
    ax.set_xlim(0.15, 0.46)
    ax.grid(alpha=0.25, axis="x", linewidth=0.5)
    fig.tight_layout()
    path = OUT / "g1_intervals.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_trust_divergence() -> Path:
    """P3.5 -- and it does not show what the plan expected.

    C3 was meant to learn that misinformation domains are unreliable. What it actually
    learns is which domains appear in *correct verifications*, which is a different thing:
    a claim citing the Congressional Budget Office is a hard claim, and a claim citing a
    tweet is often an easy one. The figure shows both halves of that.
    """
    from attacks.memory import load_misinfo_domains
    from experience.format import read_jsonl
    from experience.store import BetaTrust

    traj = read_jsonl(Path("data/trajectories/trajectories.jsonl"))
    failed = Path("data/trajectories/trajectories_failed.jsonl")
    if failed.exists():
        traj += read_jsonl(failed)
    trust = BetaTrust()
    for t in traj:
        trust.observe(t)

    misinfo = set(load_misinfo_domains(Path("data/averitec/misinfo_list.txt")))
    seen = trust.as_dict()
    # Observations beyond the prior, so "well observed" means real evidence.
    obs = {k: trust.alpha.get(k, 1.0) + trust.beta.get(k, 3.0) - 4.0 for k in seen}

    bad = [v for k, v in seen.items() if k in misinfo]
    good = [v for k, v in seen.items() if k not in misinfo]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 4.1))

    bins = [i / 20 for i in range(21)]
    ax.hist(
        good,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#0072b2",
        label=f"other publishers (n={len(good)})",
    )
    ax.hist(
        bad,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#d55e00",
        label=f"on AVeriTeC's misinformation list (n={len(bad)})",
    )
    ax.axvline(0.3, color="#444444", linestyle=":", linewidth=1.5)
    ax.annotate("trust floor", xy=(0.31, ax.get_ylim()[1] * 0.60), fontsize=8, color="#444444")
    ax.set_xlabel("Beta trust  =  alpha / (alpha + beta)")
    ax.set_ylabel("density")
    ax.set_title(
        f"C3 does not separate the two\nmeans {sum(bad) / len(bad):.3f} vs "
        f"{sum(good) / len(good):.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)

    # The ordering among domains with real evidence behind them, which is the diagnosis.
    well = sorted((k for k in seen if obs[k] >= 10), key=lambda k: seen[k])
    show = well[:7] + well[-7:]
    ys = range(len(show))
    colours = ["#d55e00" if k in misinfo else "#0072b2" for k in show]
    bx.barh(list(ys), [seen[k] for k in show], color=colours, alpha=0.9)
    bx.axvline(0.3, color="#444444", linestyle=":", linewidth=1.5)
    bx.set_yticks(list(ys))
    bx.set_yticklabels([f"{k}  ({int(obs[k])})" for k in show], fontsize=7.5)
    bx.set_xlabel("Beta trust  (number of observations in brackets)")
    bx.set_title("What it learned instead: which domains\nappear in correct verdicts", fontsize=10)
    bx.set_xlim(0, 0.85)
    bx.grid(alpha=0.25, axis="x", linewidth=0.5)

    fig.tight_layout()
    path = OUT / "trust_divergence.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    for fn in (
        fig_memory_poisoning,
        fig_adversary_inversion,
        fig_source_poisoning,
        fig_g1_intervals,
        fig_trust_divergence,
    ):
        try:
            path = fn()
        except FileNotFoundError as exc:
            print(f"  skipped {fn.__name__}: {exc}")
            continue
        print(f"  {path}  ({path.stat().st_size / 1000:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
