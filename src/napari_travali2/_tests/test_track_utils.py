import pytest

@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_find_track_successors(graph_type, make_empty_graph):
    graph = make_empty_graph(graph_type)
    graph.add_node({})
    track_id = 1
    successors_df = find_track_successors(graph, track_id, track_id_attr_key)
    assert not successors_df.is_empty()
    assert "node_id" in successors_df.columns
    assert "t" in successors_df.columns