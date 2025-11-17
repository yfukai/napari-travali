from __future__ import annotations

import numpy as np
import pytest
import tracksdata as td

from napari_travali2.actionable_tracks.action import (
    AnnotateDaughterAction,
    AnnotateTerminationAction,
    ConnectTrackAction,
    MergeLabelsAction,
    RedrawMaskAction,
)
from napari_travali2.actionable_tracks.actionable_tracks import ActionableTracks

@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_assign_tracklet_ids_assigns_consistent_ids(
    graph_type, make_empty_graph, add_node
):
    graph = make_empty_graph(graph_type)
    node_ids = [
        add_node(graph, t=frame, offset=frame * 10) for frame in range(3)
    ]
    graph.add_edge(node_ids[0], node_ids[1], {})
    graph.add_edge(node_ids[1], node_ids[2], {})
    tracks = ActionableTracks(graph)

    tracks.assign_tracklet_ids(node_ids)

    df = graph.filter(node_ids=node_ids).node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.TRACKLET_ID]
    )
    assigned_ids = df[td.DEFAULT_ATTR_KEYS.TRACKLET_ID].to_list()
    assert assigned_ids == [1, 1, 1]
    assert tracks.safe_tracklet_id >= 2


@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_redraw_mask_action_overwrites_mask_and_bbox(
    graph_type, make_sample_graph, make_mask
):
    graph, nodes_dict = make_sample_graph(graph_type)
    tracks = ActionableTracks(graph)

    node_id = nodes_dict["A"][1]
    filtered = tracks.graph.filter(node_ids=[node_id]).node_attrs(
        attr_keys=[tracks.mask_attr_name]
    )
    original_mask = filtered[tracks.mask_attr_name][0]

    action = RedrawMaskAction(
        node_id=node_id,
        new_mask=make_mask(offset=50, extra_pixel=True),
    )
    action.apply(tracks)

    updated = graph.filter(node_ids=[node_id]).node_attrs(
        attr_keys=[tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert action.old_mask == original_mask
    assert updated[tracks.mask_attr_name][0] == action.new_mask
    assert np.array_equal(
        np.array(updated[tracks.bbox_attr_name][0]), action.new_mask.bbox
    )


@pytest.mark.parametrize("reconnect_others", [False, True])
@pytest.mark.parametrize("test_param", [
        ("A2", "B3", [("A2", "A3"), ("B2", "B3")], [("B2", "A3")], 
         [[["A0", "A1", "A2", "B3"], ["A3"], ["B0", "B1"], ["B2"], ["B4", "B5"], ["C0"]] ,
          [["A0", "A1", "A2", "B3"], ["B0", "B1"], ["B2", "A3"], ["B4", "B5"], ["C0"]]]),
        ("A1", "B2", [("A1", "A2"), ("B1", "B2")], [("B1", "A2")], 
         [[["A0", "A1", "B2", "B3"], ["A2", "A3"], ["B0", "B1", "B4", "B5"], ["C0"]],
          [["A0", "A1", "B2", "B3"], ["A2", "A3"], ["B0", "B1"], ["B4", "B5"], ["C0"]]]),
        ("A0", "C0", [("A0", "A1")], [], 
         [[["A0", "C0"], ["A1", "A2", "A3"], ["B0", "B1"], ["B2", "B3"], ["B4", "B5"]],
          [["A0", "C0"], ["A1", "A2", "A3"], ["B0", "B1"], ["B2", "B3"], ["B4", "B5"]]]),
        ("C0", "B4", [("B1", "B4")], [], 
         [[["A0", "A1", "A2", "A3"], ["B0", "B1", "B2", "B3"], ["C0", "B4", "B5"]],
          [["A0", "A1", "A2", "A3"], ["B0", "B1", "B2", "B3"], ["C0", "B4", "B5"]]]),
    ])
@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_connect_track_action_links_nodes_and_reconnects_successors(
    graph_type,
    reconnect_others,
    test_param,
    make_sample_graph,
    assert_graphs_isomorphic,
    compare_tracklet_id_assignments,
):
    graph, nodes_dict = make_sample_graph(graph_type)
    def _map_node_name_to_id(name: str) -> int:
        return nodes_dict[name[0]][int(name[1])]
    node_id1 = _map_node_name_to_id(test_param[0])
    node_id2 = _map_node_name_to_id(test_param[1])
    remove_edges = [
        (_map_node_name_to_id(src), _map_node_name_to_id(tgt))
        for src, tgt in test_param[2]
    ]
    add_edges_reconnect = [
        (_map_node_name_to_id(src), _map_node_name_to_id(tgt))
        for src, tgt in test_param[3]
    ]
    tracklet_node_sets = [[[_map_node_name_to_id(name) for name in group1] for group1 in group] for group in test_param[4]]
    
    target_graph, _ = make_sample_graph(graph_type)
    tracks = ActionableTracks(graph)
    tracks.assign_tracklet_ids()
    action = ConnectTrackAction(
        node_id1=node_id1,
        node_id2=node_id2,
        reconnect_others=reconnect_others,
    )
    action.apply(tracks)

    for edge in remove_edges:
        target_graph.remove_edge(*edge)
    target_graph.add_edge(node_id1, node_id2, {})
    if reconnect_others:
        for edge in add_edges_reconnect:
            target_graph.add_edge(*edge, {})
            
    # Check if isomorphic
    assert_graphs_isomorphic(tracks.graph, target_graph)
    # Check tracklet IDs
    compare_tracklet_id_assignments(
        tracklet_node_sets[1 if reconnect_others else 0],
        tracks.graph,
    )        

@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_connect_track_action_validates_time_ordering(
    graph_type, make_empty_graph, add_node
):
    graph = make_empty_graph(graph_type)
    late = add_node(graph, t=5, offset=0)
    early = add_node(graph, t=3, offset=10)
    tracks = ActionableTracks(graph)

    action = ConnectTrackAction(
        node_id1=late,
        node_id2=early,
        reconnect_others=False,
    )

    with pytest.raises(ValueError):
        action.apply(tracks)


@pytest.mark.parametrize("graph_type", ["inmem", "sql"])    
def test_annotate_daughter_action_reuses_and_creates_nodes(
    graph_type,
    make_sample_graph,
    make_mask,
    compare_tracklet_id_assignments,
    assert_graphs_isomorphic,
):
    graph, nodes_dict = make_sample_graph(graph_type)
    
    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]

    parent = B1
    existing = A2
    new_mask = make_mask(offset=4)
    tracks = ActionableTracks(graph)
    tracks.assign_tracklet_ids()
    action = AnnotateDaughterAction(
        node_id=B1,
        daughters=[
            A2,  # existing node
            (2, new_mask) # new node
        ],
    )
    before_node_ids = set(tracks.graph.node_ids())
    daughter_node_ids = action.apply(tracks)
    new_node_id = daughter_node_ids[1]
    after_node_ids = set(tracks.graph.node_ids())
    assert before_node_ids.union({new_node_id}) == after_node_ids
    assert new_node_id not in before_node_ids

    assert tracks.graph.has_edge(parent, existing)
    assert tracks.graph.has_edge(parent, new_node_id)

    created_attrs = tracks.graph.filter(node_ids=[new_node_id]).node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.T, tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert created_attrs[td.DEFAULT_ATTR_KEYS.T][0] == 2
    assert created_attrs[tracks.mask_attr_name][0] == new_mask
    assert np.array_equal(
        np.array(created_attrs[tracks.bbox_attr_name][0]), new_mask.bbox
    )

    compare_tracklet_id_assignments(
        [[A0, A1], [A2, A3], [B0, B1], [B2, B3], [B4, B5], [new_node_id], [C1]],
        tracks.graph,
    )

    target_graph, _ = make_sample_graph(graph_type)
    target_graph.add_edge(parent, existing, {})
    target_graph.remove_edge(A1, A2)
    new_node_id_target = target_graph.add_node(
        {
            td.DEFAULT_ATTR_KEYS.T: 2,
            tracks.mask_attr_name: new_mask,
            tracks.bbox_attr_name: new_mask.bbox,
            tracks.tracklet_id_attr_name: -1,
            "termination_annotation": "",
            "verified": False,
        }
    )
    target_graph.add_edge(parent, new_node_id_target, {})
    target_graph.remove_edge(B1, B2)
    target_graph.remove_edge(B1, B4)
    assert_graphs_isomorphic(tracks.graph, target_graph)


@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_merge_labels_action_unifies_masks_and_removes_source_node(
    graph_type, make_sample_graph, compare_tracklet_id_assignments
):
    graph, nodes_dict = make_sample_graph(graph_type)

    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]
    
    tracks = ActionableTracks(graph)
    masks = graph.filter(node_ids=[A2, B2, B4]).node_attrs(
        attr_keys=[tracks.mask_attr_name]
    )
    A2_mask = masks[tracks.mask_attr_name][0]
    B2_mask = masks[tracks.mask_attr_name][1]
    B4_mask = masks[tracks.mask_attr_name][2]
    
    tracks.assign_tracklet_ids()
    action = MergeLabelsAction(
        node_id_merged=A2,
        node_id_target=B2,
    )
    action.apply(tracks)

    remaining_ids = list(tracks.graph.node_ids())
    assert A2 not in remaining_ids
    filtered = tracks.graph.filter(node_ids=[B2]).node_attrs(
        attr_keys=[tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    expected_mask = A2_mask | B2_mask
    assert filtered[tracks.mask_attr_name][0] == expected_mask
    assert np.array_equal(
        np.array(filtered[tracks.bbox_attr_name][0]), expected_mask.bbox
    )
    compare_tracklet_id_assignments(
        [[A0, A1], [A3], [B0, B1], [B2, B3], [B4, B5], [C1]],
        tracks.graph,
    )

@pytest.mark.parametrize(
    "node_name, delete_successor_tracklet, expected_remaining_node_names, expected_groups_names",
    [
        (
            "B1",
            True,
            {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4", "B5", "C0"},
            [["A0", "A1", "A2", "A3"], ["B0", "B1"], ["B2", "B3"], ["B4", "B5"], ["C0"]],
        ),
        (
            "B1",
            False,
            {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4", "B5", "C0"},
            [["A0", "A1", "A2", "A3"], ["B0", "B1"], ["B2", "B3"], ["B4", "B5"], ["C0"]],
        ),
        (
            "B2",
            True,
            {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B4", "B5", "C0"},
            [["A0", "A1", "A2", "A3"], ["B0", "B1"], ["B2"], ["B4", "B5"], ["C0"]],
        ),
        (
            "B2",
            False,
            {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4", "B5", "C0"},
            [["A0", "A1", "A2", "A3"], ["B0", "B1"], ["B2"], ["B3"], ["B4", "B5"], ["C0"]],
        ),
    ],
)
@pytest.mark.parametrize("graph_type", ["inmem", "sql"])
def test_annotate_termination_action_updates_annotation_and_handles_successors(
    node_name,
    delete_successor_tracklet,
    expected_remaining_node_names,
    expected_groups_names,
    graph_type,
    make_sample_graph,
    compare_tracklet_id_assignments,
):
    graph, nodes_dict = make_sample_graph(graph_type)
    def _map_node_name_to_id(name: str) -> int:
        return nodes_dict[name[0]][int(name[1])]
    node_id = _map_node_name_to_id(node_name)
    expected_remaining_node_ids = {_map_node_name_to_id(name) for name in expected_remaining_node_names}
    expected_groups = [
        [_map_node_name_to_id(name) for name in group]
        for group in expected_groups_names
    ]
    
    tracks = ActionableTracks(graph)
    tracks.assign_tracklet_ids()

    action = AnnotateTerminationAction(
        node_id=node_id,
        termination_annotation="finished",
        delete_successor_tracklet=delete_successor_tracklet,
    )
    action.apply(tracks)

    attrs = tracks.graph.filter(node_ids=[node_id]).node_attrs(
        attr_keys=[tracks.termination_annotation_attr_name]
    )
    assert attrs[tracks.termination_annotation_attr_name][0] == "finished"
    assert len(tracks.graph.successors(node_id)) == 0

    remaining_node_ids = set(tracks.graph.node_ids())
    assert remaining_node_ids == expected_remaining_node_ids

    compare_tracklet_id_assignments(expected_groups, tracks.graph)
