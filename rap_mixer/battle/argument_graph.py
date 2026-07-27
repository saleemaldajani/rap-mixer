from __future__ import annotations

import re

from rap_mixer.schemas.battle import ArgumentEdge, ArgumentNode


def build_argument_graph(bars: list[str]) -> tuple[list[ArgumentNode], list[ArgumentEdge]]:
    nodes, edges = [], []
    for i, text in enumerate(bars, 1):
        low = text.lower()
        kind = "Attack" if any(x in low for x in ("you ", "your ", "can't", "weak")) else "Boast"
        if any(x in low for x in ("because", "proof", "show")):
            kind = "Evidence"
        elif any(x in low for x in ("but ", "yet ", "though")):
            kind = "Claim"
        elif any(x in low for x in ("like ", "laugh", "joke")):
            kind = "Joke"
        clean = re.sub(r"\s+", " ", text).strip()
        nodes.append(ArgumentNode(id=f"N{i}", bar_id=f"H{i}", kind=kind, text=clean))
        if i > 1:
            edges.append(ArgumentEdge(source=f"N{i-1}", target=f"N{i}", relation="Supports"))
    return nodes, edges


def graph_rows(nodes, edges) -> list[dict]:
    edge_by_target = {x.target: f"{x.relation} {x.source}" for x in edges}
    return [{"node": n.id, "human_bar": n.bar_id, "type": n.kind, "text": n.text,
             "relationship": edge_by_target.get(n.id, "root")} for n in nodes]

