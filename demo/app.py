"""Gradio demo: verify a claim, then switch the defences off and watch it break (P4.7).

The demo exists to make one thing visible that a results table cannot: that the same
claim, the same evidence and the same model produce a different verdict depending on what
is in the experience store and which filters are on. Attack mode injects fabricated
trajectories into a *copy* of the store and re-runs, side by side.

**Everything replays from cache.** The project's hardware rule is that the Mac is dev-only
and never runs bulk inference, so the demo defaults to claims that have already been
verified during the experiments -- each one is a cache hit and returns immediately. A
claim that was never run needs a live endpoint, and the UI says so rather than appearing
to hang.

**Nothing fabricated is written to disk.** The poisoned store is assembled in memory for
one request and discarded (ethics rule 10).

    python -m demo.app                 # then open the printed URL
    python -m demo.app --share         # temporary public link, for a remote viva
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from agent.crag import crag
from attacks.memory import (
    FLIP,
    Target,
    load_misinfo_domains,
    poison_store,
    trusted_domains,
)
from core.config import RunConfig
from core.llm import LLMClient
from core.paths import DATA_DIR, RESULTS_DIR
from eval.datasets import Claim, load_averitec
from evaluator.score import load_scorer
from experience.format import read_jsonl
from experience.store import BetaTrust, ExperienceStore
from retrieval.blocklist import domain_of
from retrieval.store import open_store

DEV = DATA_DIR / "averitec" / "dev.json"
POOL = DATA_DIR / "averitec" / "pool_dev.jsonl"
TRAJECTORIES = DATA_DIR / "trajectories" / "trajectories.jsonl"
MISINFO = DATA_DIR / "averitec" / "misinfo_list.txt"

VERDICT_COLOUR = {
    "Supported": "#1a7f37",
    "Refuted": "#cf222e",
    "Conflicting evidence": "#9a6700",
    "Not enough evidence": "#57606a",
}


class Backend:
    """Everything the demo needs, loaded once."""

    def __init__(self) -> None:
        missing = [p for p in (DEV, POOL) if not p.exists()]
        if missing:
            raise SystemExit(
                "the demo needs the AVeriTeC dev split and evidence pool:\n  "
                + "\n  ".join(str(p) for p in missing)
            )
        self.claims = {str(c.id): c for c in load_averitec(DEV, "dev")}
        self.source = open_store(POOL)
        self.llm = LLMClient()

        hits = sorted(Path(RESULTS_DIR).glob("*_t5-evaluator/model/config.json"))
        self.evaluator = hits[-1].parent if hits else None
        self.scorer = load_scorer(self.evaluator)

        self.past = read_jsonl(TRAJECTORIES) if TRAJECTORIES.exists() else []
        self.trust = BetaTrust()
        for t in self.past:
            self.trust.observe(t)
        self.domains = load_misinfo_domains(MISINFO)

    def store_for(self, claim: Claim, rate: float, adaptive: bool) -> ExperienceStore | None:
        """A store, optionally poisoned against this claim. In memory only."""
        if not self.past:
            return None
        trajectories = self.past
        if rate > 0:
            # FLIP, not a local rule: the demo must attack the same verdict the
            # experiments measured. A hand-written rule here disagreed with it for
            # "Conflicting evidence" (FLIP sends it to Supported, the rule sent it to
            # Refuted), so the demo was showing a different attack from the paper.
            gold = claim.gold_label or "Refuted"
            target = Target(
                claim_id=str(claim.id),
                claim=claim.claim,
                gold_label=gold,
                target_verdict=FLIP[gold],
            )
            domains = trusted_domains(self.past, self.trust) if adaptive else self.domains
            trajectories, _ = poison_store(self.past, [target], rate, seed=13, domains=domains)
        return ExperienceStore(
            trajectories,
            scorer=self.scorer,
            trust=self.trust,
            gate_threshold=RunConfig().gate_threshold,
            trust_floor=RunConfig().trust_floor,
        )

    def run(
        self, claim: Claim, *, memory: bool, gate: bool, trust: bool, rate: float, adaptive: bool
    ) -> tuple[Any, list]:
        cfg = RunConfig(
            split="demo",
            experience_retrieval=memory,
            gate_experience=memory and gate,
            source_trust=trust,
        )
        store = self.store_for(claim, rate, adaptive) if memory or trust else None
        trajectory = crag(
            claim,
            self.llm,
            self.source,
            self.scorer,
            cfg,
            memory=store if memory else None,
            trust=self.trust if trust else None,
        )

        # `crag` records only the ids of the trajectories that reached the prompt, which
        # is all a results table needs. A demo needs the records themselves -- which were
        # fabricated, what each scored at the gate, and why any was dropped -- so the
        # store is queried again for display. Retrieval is deterministic and the
        # evaluator is cached, so this returns what the pipeline actually saw.
        considered: list = []
        if store is not None and memory:
            considered = store.retrieve(
                claim.claim,
                claim_id=str(claim.id),
                k=cfg.k_trajectories,
                gate=cfg.gate_experience,
                use_trust=cfg.source_trust,
                return_dropped=True,
            )
        return trajectory, considered


CSS = """
.vm-wrap { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.vm-card { border: 1px solid var(--border-color-primary); border-radius: 10px;
           padding: 1.05rem 1.15rem 1.2rem; background: var(--background-fill-primary); }
.vm-head { font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
           color: var(--body-text-color-subdued); margin-bottom: .5rem; }
.vm-verdict { font-size: 2rem; font-weight: 700; line-height: 1.1; margin: .1rem 0 .35rem; }
.vm-gold { font-size: .85rem; color: var(--body-text-color-subdued); margin-bottom: .75rem; }
.vm-why { font-size: .9rem; line-height: 1.5; margin-bottom: 1rem; }
.vm-sec { font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
          color: var(--body-text-color-subdued); margin: 1rem 0 .45rem; }
.vm-ev { border-left: 3px solid var(--border-color-primary); padding: .15rem 0 .15rem .6rem;
         margin: .35rem 0; font-size: .85rem; line-height: 1.45; }
.vm-g { font-family: ui-monospace, Menlo, monospace; font-size: .8rem;
        font-variant-numeric: tabular-nums; }
.vm-rec { border: 1px solid var(--border-color-primary); border-radius: 7px;
          padding: .6rem .75rem; margin: .45rem 0; font-size: .85rem; line-height: 1.45; }
.vm-rec.fake { border-color: #cf222e; background: rgba(207,34,46,.07); }
.vm-rec.gone { opacity: .6; }
.vm-tag { display: inline-block; font-family: ui-monospace, Menlo, monospace;
          font-size: .66rem; letter-spacing: .05em; padding: .1em .45em;
          border-radius: 3px; vertical-align: middle; }
.vm-tag.fake { background: #cf222e; color: #fff; }
.vm-tag.real { background: var(--background-fill-secondary);
               color: var(--body-text-color-subdued); }
.vm-tag.gone { background: #1a7f37; color: #fff; }
.vm-banner { border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .9rem;
             font-size: 1.02rem; line-height: 1.45; }
.vm-banner.landed { background: rgba(207,34,46,.12); border: 1px solid #cf222e; }
.vm-banner.held   { background: rgba(26,127,55,.12); border: 1px solid #1a7f37; }
.vm-banner.quiet  { background: var(--background-fill-secondary);
                    border: 1px solid var(--border-color-primary); }
.vm-arrow { font-size: 1.45rem; font-weight: 700; opacity: .6; }
"""


def _verdict_colour(v: str) -> str:
    return VERDICT_COLOUR.get(v, "#57606a")


def _is_fake(t: Any) -> bool:
    """A fabricated record carries the flag, and also names itself in its claim id."""
    return bool(getattr(t, "poisoned", False)) or str(t.claim_id or "").startswith("poison-")


def _records(considered: list) -> str:
    """The retrieved past verifications -- the attack surface itself.

    Listing bare ids told a viewer nothing. What matters is which of these records is a
    fabrication, what verdict it asserts, and, where a defence removed it, which defence
    and on what number. That last part is the only place a viewer can watch C2 and C3
    actually working rather than infer it from a verdict that did not move.
    """
    if not considered:
        return "<div class='vm-ev'>none retrieved</div>"
    out = []
    for r in considered:
        t = r.trajectory
        fake = _is_fake(t)
        gone = not r.kept
        cls = "vm-rec" + (" fake" if fake else "") + (" gone" if gone else "")
        tag = (
            "<span class='vm-tag fake'>FABRICATED</span>"
            if fake
            else "<span class='vm-tag real'>genuine</span>"
        )
        if gone:
            why = (r.dropped_by or "a filter").upper()
            detail = ""
            if r.dropped_by == "trust" and r.min_trust is not None:
                detail = f" &middot; trust {r.min_trust:.2f} under floor"
            elif r.dropped_by == "gate" and r.gate_score is not None:
                detail = f" &middot; gate {r.gate_score:+.2f}"
            tag += f" <span class='vm-tag gone'>BLOCKED BY {html.escape(why)}{detail}</span>"
        colour = _verdict_colour(t.verdict)
        domains = ", ".join(sorted(t.domains)[:3]) or "no sources"
        out.append(
            f"<div class='{cls}'>{tag}<br>"
            f"<span style='color:{colour};font-weight:600'>{html.escape(t.verdict)}</span>"
            f" &mdash; <span style='opacity:.85'>{html.escape(t.claim[:118])}</span><br>"
            f"<span style='font-size:.78rem;opacity:.7'>{html.escape(domains)}</span></div>"
        )
    return "".join(out)


def _panel(title: str, t: Any, considered: list, trust: BetaTrust | None) -> str:
    colour = _verdict_colour(t.verdict)
    mark = ""
    if t.gold_label:
        ok = t.correct
        mark = (
            f"<span style='color:{'#1a7f37' if ok else '#cf222e'};font-weight:600'>"
            f"{'correct' if ok else 'WRONG'}</span>"
        )
    rows = []
    for e in t.evidence[:4]:
        dom = domain_of(e.url) or "?"
        tr = f" &middot; trust {trust.trust(dom):.2f}" if trust else ""
        rows.append(
            f"<div class='vm-ev'><span class='vm-g'>{e.grade:+.3f}</span> "
            f"<b>{html.escape(dom)}</b>"
            f"<span style='opacity:.65;font-size:.8rem'>{tr}</span><br>"
            f"{html.escape(e.text[:165])}</div>"
        )
    n_fake = sum(1 for r in considered if _is_fake(r.trajectory) and r.kept)
    fake_note = f" &middot; {n_fake} fabricated reached the prompt" if n_fake else ""
    evidence_html = "".join(rows) or "<div class='vm-ev'>none</div>"
    return (
        "<div class='vm-card vm-wrap'>"
        f"<div class='vm-head'>{html.escape(title)}</div>"
        f"<div class='vm-verdict' style='color:{colour}'>{html.escape(t.verdict)}</div>"
        f"<div class='vm-gold'>gold: {html.escape(t.gold_label or 'n/a')} {mark}</div>"
        f"<div class='vm-why'>{html.escape(t.rationale or '')}</div>"
        "<div class='vm-sec'>Evidence it used</div>"
        f"{evidence_html}"
        f"<div class='vm-sec'>Past verifications retrieved{fake_note}</div>"
        f"{_records(considered)}"
        "</div>"
    )


def _banner(claim: Claim, clean: Any, attacked: Any, rate: int) -> str:
    """What changed, in one line, legible from the back of a room."""
    if attacked is None or rate == 0:
        return (
            "<div class='vm-banner quiet vm-wrap'>Store is clean. Raise the poisoned "
            "share to inject fabricated records against this claim.</div>"
        )
    wanted = FLIP.get(claim.gold_label or "", "?")
    landed = attacked.verdict == wanted
    changed = attacked.verdict != clean.verdict
    cls = "landed" if landed else ("held" if changed else "quiet")
    wanted_html = f"<b style='color:{_verdict_colour(wanted)}'>{html.escape(wanted)}</b>"
    if landed:
        msg = f"<b>Attack succeeded.</b> The adversary wanted {wanted_html} and got it."
    elif changed:
        msg = (
            f"<b>Attack blocked.</b> It wanted {wanted_html}; the verdict moved, but not "
            "to the adversary's answer."
        )
    else:
        msg = f"<b>No effect.</b> The verdict is unchanged; the adversary wanted {wanted_html}."
    return (
        f"<div class='vm-banner {cls} vm-wrap'>"
        f"<span style='color:{_verdict_colour(clean.verdict)};font-weight:700'>"
        f"{html.escape(clean.verdict)}</span> "
        "<span class='vm-arrow'>&rarr;</span> "
        f"<span style='color:{_verdict_colour(attacked.verdict)};font-weight:700'>"
        f"{html.escape(attacked.verdict)}</span><br>{msg}</div>"
    )


HOW_TO = """
1. Pick a **★** claim and press **Verify**. The clean store gives the right answer.
2. Drag the slider to **20%**. Fabricated records appear in the right-hand panel,
   outlined in red, each showing the verdict it asserts and the domains it cites.
3. Untick **C2** and **C3**. The fabrications now reach the prompt and the verdict
   flips — the banner reads *Attack succeeded*.
4. Tick them back on. Blocked records show **which defence removed them, and on what
   number**.
5. Now tick **Adaptive adversary**. It cites the domains the agent trusts most, so C3
   stops blocking — and only C2 is left holding the line.

*Fabricated records are assembled in memory for one request and discarded. Nothing
fabricated is written to disk.*
"""


def build(backend: Backend):
    import gradio as gr  # noqa: PLC0415

    # Verified to show the full sequence: correct when clean, flipped when attacked
    # undefended, blocked when the defences are on. Re-measured 2026-11-21 by
    # `python -m demo.find_cases --limit 70`, which shortlisted 8 full cases out of the 35
    # it evaluated. That shortlist is written to `demo/demo_cases.json`, which is
    # gitignored -- so these ids are kept here, in code, rather than read from it.
    #
    # These three recover the *correct* verdict rather than retreating to an abstention,
    # which is the stronger thing to show. 75 leads because it is innocuous to project.
    # Claim "0" used to be starred here and is not a demo case at all -- it never flips,
    # so starring it invited clicking a claim mid-demo and watching nothing happen.
    SUGGESTED = ("75", "219", "182")
    labelled = [(cid, c) for cid, c in backend.claims.items() if c.gold_label]
    ordered = [(cid, backend.claims[cid]) for cid in SUGGESTED if cid in backend.claims]
    ordered += [(cid, c) for cid, c in labelled if cid not in SUGGESTED][:400]
    options = [f"{'★ ' if cid in SUGGESTED else ''}{cid} — {c.claim[:86]}" for cid, c in ordered]

    def verify(selection: str, rate_pct: int, adaptive: bool, gate: bool, trust: bool):
        if not selection:
            return "", "", ""
        cid = selection.replace("★ ", "").split(" — ")[0]
        claim = backend.claims.get(cid)
        if claim is None:
            return f"<p>unknown claim {html.escape(cid)}</p>", "", ""

        clean, clean_rec = backend.run(
            claim, memory=True, gate=gate, trust=trust, rate=0.0, adaptive=False
        )
        left = _panel("Clean store", clean, clean_rec, backend.trust if trust else None)
        if rate_pct == 0:
            return _banner(claim, clean, None, 0), left, ""

        attacked, att_rec = backend.run(
            claim,
            memory=True,
            gate=gate,
            trust=trust,
            rate=rate_pct / 100.0,
            adaptive=adaptive,
        )
        title = f"{rate_pct}% poisoned · {'adaptive' if adaptive else 'naive'} adversary"
        right = _panel(title, attacked, att_rec, backend.trust if trust else None)
        return _banner(claim, clean, attacked, rate_pct), left, right

    # Gradio 6 moved css and theme from Blocks to launch(); passing them here is
    # accepted with a warning and then ignored, which silently drops the styling.
    with gr.Blocks(title="VeriMem") as app:
        gr.Markdown(
            "# VeriMem\n"
            "### Fact-check a claim, poison the agent's memory, watch the verdict move.\n"
            "Claims marked ★ are verified to show the full sequence. Everything replays "
            "from cache, so results are immediate."
        )
        if backend.evaluator is None:
            gr.Markdown(
                "> **No trained evaluator found.** The relevance grades below are a "
                "lexical stub, not the trained model."
            )

        with gr.Row():
            pick = gr.Dropdown(
                options, label="Claim", value=options[0] if options else None, scale=5
            )
            go = gr.Button("Verify", variant="primary", scale=1)

        with gr.Row():
            rate = gr.Slider(
                0, 20, value=0, step=5, label="Poisoned share of the memory (%)", scale=3
            )
            adaptive = gr.Checkbox(
                False,
                label="Adaptive adversary",
                info="cites the domains the agent already trusts",
                scale=2,
            )
        with gr.Row():
            gate = gr.Checkbox(
                True,
                label="C2 — evaluator gating",
                info="does a record's evidence support its own verdict?",
            )
            trust = gr.Checkbox(
                True,
                label="C3 — source trust",
                info="drop records citing low-trust publishers",
            )

        banner = gr.HTML()
        with gr.Row(equal_height=False):
            left = gr.HTML()
            right = gr.HTML()

        inputs = [pick, rate, adaptive, gate, trust]
        outputs = [banner, left, right]
        go.click(verify, inputs, outputs)
        for control in (pick, rate, adaptive, gate, trust):
            control.change(verify, inputs, outputs)

        with gr.Accordion("How to run this as a demonstration", open=False):
            gr.Markdown(HOW_TO)
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m demo.app")
    p.add_argument("--share", action="store_true", help="temporary public link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    try:
        import gradio as gr  # noqa: PLC0415
    except ImportError:
        raise SystemExit("gradio is not installed:  pip install gradio") from None

    print("[verimem] loading...")
    backend = Backend()
    print(f"[verimem] {len(backend.claims)} claims, {len(backend.past)} past verifications")
    build(backend).launch(
        share=args.share,
        server_port=args.port,
        css=CSS,
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
