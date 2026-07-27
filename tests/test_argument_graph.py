from rap_mixer.battle.argument_graph import build_argument_graph


def test_argument_graph_keeps_human_bar_ids():
    nodes, edges = build_argument_graph(["You claim your flow is perfect", "But the timing needs proof"])
    assert [x.bar_id for x in nodes] == ["H1", "H2"]
    assert edges[0].source == "N1" and edges[0].target == "N2"

