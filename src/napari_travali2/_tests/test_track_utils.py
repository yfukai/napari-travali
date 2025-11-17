def test_find_track_successors(tracks_graph_with_two_tracks):
    graph, track_id_attr_key = tracks_graph_with_two_tracks
    track_id = 1
    successors_df = find_track_successors(graph, track_id, track_id_attr_key)
    assert not successors_df.is_empty()
    assert "node_id" in successors_df.columns
    assert "t" in successors_df.columns