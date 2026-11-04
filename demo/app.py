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
    ) -> Any:
        cfg = RunConfig(
            split="demo",
            experience_retrieval=memory,
            gate_experience=memory and gate,
            source_trust=trust,
        )
        store = self.store_for(claim, rate, adaptive) if memory or trust else None
        return crag(
            claim,
            self.llm,
            self.source,
            self.scorer,
            cfg,
            memory=store if memory else None,
            trust=self.trust if trust else None,
        )


def _panel(title: str, t: Any, trust: BetaTrust | None) -> str:
    colour = VERDICT_COLOUR.get(t.verdict, "#57606a")
    mark = ""
    if t.gold_label:
        ok = t.correct
        mark = (
            f"<span style='color:{'#1a7f37' if ok else '#cf222e'};font-weight:600'>"
            f"{'correct' if ok else 'WRONG'}</span>"
        )
    rows = []
    for e in t.evidence[:5]:
        dom = domain_of(e.url) or "?"
        tr = f" · trust {trust.trust(dom):.2f}" if trust else ""
        rows.append(
            f"<div style='margin:.4em 0;padding-left:.6em;border-left:3px solid #d0d7de'>"
            f"<code style='font-size:.85em'>{e.grade:+.3f}</code> "
            f"<b style='font-size:.85em'>{html.escape(dom)}</b>"
            f"<span style='color:#57606a;font-size:.8em'>{tr}</span><br>"
            f"<span style='font-size:.85em'>{html.escape(e.text[:190])}</span></div>"
        )
    used = t.metadata.get("retrieved_ids") or []
    return f"""
<div style='border:1px solid #d0d7de;border-radius:8px;padding:1em'>
  <div style='font-size:.8em;color:#57606a;text-transform:uppercase'>{html.escape(title)}</div>
  <div style='font-size:1.5em;font-weight:700;color:{colour};margin:.2em 0'>
    {html.escape(t.verdict)}</div>
  <div style='font-size:.85em;margin-bottom:.6em'>
    gold: {html.escape(t.gold_label or "n/a")} {mark}</div>
  <div style='font-size:.9em;margin-bottom:.8em'>{html.escape(t.rationale or "")}</div>
  <div style='font-size:.8em;color:#57606a'>EVIDENCE</div>
  {"".join(rows) or "<i>none</i>"}
  <div style='font-size:.8em;color:#57606a;margin-top:.8em'>
    PAST VERIFICATIONS USED: {len(used)} {html.escape(", ".join(map(str, used[:4])))}</div>
</div>"""


def build(backend: Backend):
    import gradio as gr  # noqa: PLC0415

    options = [
        f"{cid} — {c.claim[:90]}" for cid, c in list(backend.claims.items())[:400] if c.gold_label
    ]

    def verify(selection: str, rate_pct: int, adaptive: bool, gate: bool, trust: bool):
        if not selection:
            return "", ""
        cid = selection.split(" — ")[0]
        claim = backend.claims.get(cid)
        if claim is None:
            return f"<p>unknown claim id {html.escape(cid)}</p>", ""

        clean = backend.run(claim, memory=True, gate=gate, trust=trust, rate=0.0, adaptive=False)
        left = _panel("clean store", clean, backend.trust if trust else None)
        if rate_pct == 0:
            return left, (
                "<div style='padding:1em;color:#57606a'>Raise the poison rate to inject "
                "fabricated trajectories against this claim.</div>"
            )
        attacked = backend.run(
            claim,
            memory=True,
            gate=gate,
            trust=trust,
            rate=rate_pct / 100.0,
            adaptive=adaptive,
        )
        wanted = FLIP.get(claim.gold_label or "", "?")
        landed = "ATTACK SUCCEEDED" if attacked.verdict == wanted else "attack did not land"
        title = (
            f"{rate_pct}% poisoned · {'adaptive' if adaptive else 'naive'} adversary · "
            f"wants {wanted} · {landed}"
        )
        return left, _panel(title, attacked, backend.trust if trust else None)

    with gr.Blocks(title="VeriMem") as app:
        gr.Markdown(
            "# VeriMem\n"
            "Fact-check a claim, then poison the agent's memory and watch the verdict move.\n\n"
            "*Claims already verified during the experiments replay from cache and return "
            "immediately. A claim that was never run needs a live endpoint.*"
        )
        if backend.evaluator is None:
            gr.Markdown(
                "> **No trained evaluator found.** Relevance grades below are a lexical "
                "stub, not the trained model."
            )
        with gr.Row():
            pick = gr.Dropdown(
                options, label="claim", value=options[0] if options else None, scale=4
            )
            go = gr.Button("verify", variant="primary", scale=1)
        with gr.Row():
            rate = gr.Slider(0, 20, value=0, step=5, label="poisoned share of the store (%)")
            adaptive = gr.Checkbox(False, label="adaptive adversary (cites trusted domains)")
        with gr.Row():
            gate = gr.Checkbox(True, label="C2 — evaluator gating")
            trust = gr.Checkbox(True, label="C3 — source trust")
        with gr.Row():
            left = gr.HTML()
            right = gr.HTML()

        inputs = [pick, rate, adaptive, gate, trust]
        go.click(verify, inputs, [left, right])
        for control in (rate, adaptive, gate, trust):
            control.change(verify, inputs, [left, right])

        gr.Markdown(
            "**Try this:** pick a claim, set poisoning to 20% with the *naive* adversary — "
            "the defences hold. Now tick *adaptive*: the attacker cites the domains the "
            "agent trusts most, C3 is defeated, and only C2 is left. Untick C2 as well and "
            "the attack lands.\n\n"
            "Fabricated trajectories are assembled in memory for one request and discarded; "
            "nothing fabricated is written to disk."
        )
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m demo.app")
    p.add_argument("--share", action="store_true", help="temporary public link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    try:
        import gradio  # noqa: F401, PLC0415
    except ImportError:
        raise SystemExit("gradio is not installed:  pip install gradio") from None

    print("[verimem] loading...")
    backend = Backend()
    print(f"[verimem] {len(backend.claims)} claims, {len(backend.past)} past verifications")
    build(backend).launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
