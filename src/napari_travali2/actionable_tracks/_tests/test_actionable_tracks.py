from __future__ import annotations

import numpy as np
import pytest
from copy import deepcopy
import tracksdata as td

from napari_travali2.actionable_tracks.action import (
    AnnotateDaughterAction,
    AnnotateTerminationAction,
    ConnectTrackAction,
    MergeLabelsAction,
    RedrawMaskAction,
)
from napari_travali2.actionable_tracks.actionable_tracks import ActionableTracks

def _make_empty_graph() -> td.graph.RustWorkXGraph:
    graph = td.graph.RustWorkXGraph()
    for key, default in [
        (td.DEFAULT_ATTR_KEYS.MASK, None),
        (td.DEFAULT_ATTR_KEYS.BBOX, None),
        (td.DEFAULT_ATTR_KEYS.TRACKLET_ID, -1),
        ("termination_annotation", ""),
    ]:
        graph.add_node_attr_key(key, default)
    return graph

def _make_mask(offset: int, extra_pixel: bool = False) -> td.nodes.Mask:
    mask_array = np.zeros((2, 2), dtype=bool)
    mask_array[0, 0] = True
    if extra_pixel:
        mask_array[1, 1] = True
    bbox = np.array(
        [offset, offset, offset + mask_array.shape[0], offset + mask_array.shape[1]]
    )
    return td.nodes.Mask(mask_array, bbox)


def _add_node(
    graph: td.graph.InMemoryGraph,
    *,
    t: int,
    offset: int,
    mask: td.nodes.Mask | None = None,
) -> int:
    mask = mask or _make_mask(offset)
    return graph.add_node(
        {
            td.DEFAULT_ATTR_KEYS.T: t,
            td.DEFAULT_ATTR_KEYS.MASK: mask,
            td.DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            td.DEFAULT_ATTR_KEYS.TRACKLET_ID: -1,
            "termination_annotation": "",
        }
    )

def _assert_graphs_isomorphic(
    graph1: td.graph.BaseGraph, graph2: td.graph.BaseGraph
) -> bool:
    """Check if two graphs are isomorphic, based on node IDs."""

    edge_df1 = graph1.edge_attrs(
        attr_keys=[]
    ).sort([td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    edge_df2 = graph2.edge_attrs(
        attr_keys=[]
    ).sort([td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    edge_df1 = edge_df1.select(
        [td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    )
    edge_df2 = edge_df2.select(
        [td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    )
    if len(edge_df1) != len(edge_df2):
        assert False
    assert edge_df1.equals(edge_df2)

@pytest.fixture
def sample_graph():
    """ Create a sample tracks graph for testing.

    Graph:
      - A: linear chain A0 -> A1 -> A2 -> A3
      - B: B0 -> B1 with B1 -> {B2, B4}, and B2 -> B3, B4 -> B5
      - C: C1 (at time 1)

    """

    graph = _make_empty_graph()

    # Build graph components
    # A chain
    A0 = _add_node(graph, t=0, offset=0)
    A1 = _add_node(graph, t=1, offset=1)
    A2 = _add_node(graph, t=2, offset=2)
    A3 = _add_node(graph, t=3, offset=3)
    graph.add_edge(A0, A1, {})
    graph.add_edge(A1, A2, {})
    graph.add_edge(A2, A3, {})

    # B branched with parent and children
    B0 = _add_node(graph, t=0, offset=1)
    B1 = _add_node(graph, t=1, offset=2)
    B2 = _add_node(graph, t=2, offset=3)
    B4 = _add_node(graph, t=2, offset=4)
    B3 = _add_node(graph, t=3, offset=5)
    B5 = _add_node(graph, t=3, offset=6)
    graph.add_edge(B0, B1, {})
    graph.add_edge(B1, B2, {})
    graph.add_edge(B1, B4, {})
    graph.add_edge(B2, B3, {})
    graph.add_edge(B4, B5, {})
    C1 = _add_node(graph, t=1, offset=10)

    return graph, {
        "A": [A0, A1, A2, A3],
        "B": [B0, B1, B2, B3, B4, B5],
        "C": [C1],
    }   


def test_assign_tracklet_ids_assigns_consistent_ids():
    graph = _make_empty_graph()
    node_ids = [
        _add_node(graph, t=frame, offset=frame * 10) for frame in range(3)
    ]
    graph.add_edge(node_ids[0], node_ids[1], {})
    graph.add_edge(node_ids[1], node_ids[2], {})
    tracks = ActionableTracks(graph)

    tracks.assign_tracklet_ids(node_ids)

    df = graph.filter(node_ids=node_ids).node_attrs(
        [td.DEFAULT_ATTR_KEYS.TRACKLET_ID]
    )
    assigned_ids = df[td.DEFAULT_ATTR_KEYS.TRACKLET_ID].to_list()
    assert assigned_ids == [1, 1, 1]
    assert tracks.safe_tracklet_id >= 2


def test_redraw_mask_action_overwrites_mask_and_bbox(sample_graph):
    graph, nodes_dict = sample_graph
    tracks = ActionableTracks(graph)

    node_id = nodes_dict["A"][1]
    filtered = tracks.graph.filter(node_ids=[node_id]).node_attrs(
        [tracks.mask_attr_name]
    )
    original_mask = filtered[tracks.mask_attr_name][0]

    action = RedrawMaskAction(
        node_id=node_id,
        new_mask=_make_mask(offset=50, extra_pixel=True)
    )
    action.apply(tracks)

    updated = graph.filter(node_ids=[node_id]).node_attrs(
        [tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert action.old_mask == original_mask
    assert updated[tracks.mask_attr_name][0] == action.new_mask
    assert np.array_equal(
        np.array(updated[tracks.bbox_attr_name][0]), action.new_mask.bbox
    )

    

def _compare_tracklet_id_assignments(expected_node_sets, graph_backend: td.graph.BaseGraph):
    """Compare tracklet ID assignments in the graph backend to expected node sets. Copied from tracksdata tests."""
    ids_df = graph_backend.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.TRACKLET_ID])
    ids_map = dict(
        zip(
            ids_df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            ids_df[td.DEFAULT_ATTR_KEYS.TRACKLET_ID].to_list(),
            strict=True,
        )
    )
    assigned = {}
    for node_id, tracklet_id in ids_map.items():
        if tracklet_id == -1:
            continue
        if tracklet_id not in assigned:
            assigned[tracklet_id] = []
        assigned[tracklet_id].append(node_id)
    assigned = {frozenset(group) for group in assigned.values()}
    expected = {frozenset(group) for group in expected_node_sets}
    assert assigned == expected


@pytest.mark.parametrize("reconnect_others", [False, True])
def test_connect_track_action_links_nodes_and_reconnects_successors(sample_graph, reconnect_others):
    graph, nodes_dict = sample_graph
    
    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]

    for (node_id1, node_id2, remove_edges, add_edges_reconnect, tracklet_node_sets) in [
        (A2, B3, [(A2, A3), (B2, B3)], [(B2, A3)], 
         [[[A0, A1, A2, B3], [A3], [B0, B1], [B2], [B4, B5], [C1]] ,
          [[A0, A1, A2, B3], [B0, B1], [B2, A3], [B4, B5], [C1]]]),
        (A1, B2, [(A1, A2), (B1, B2)], [(B1, A2)], 
         [[[A0, A1, B2, B3], [A2, A3], [B0, B1, B4, B5], [C1]],
          [[A0, A1, B2, B3], [A2, A3], [B0, B1], [B4, B5], [C1]]]),
        (A0, C1, [(A0, A1)], [], 
         [[[A0, C1], [A1, A2, A3], [B0, B1], [B2, B3], [B4, B5]],
          [[A0, C1], [A1, A2, A3], [B0, B1], [B2, B3], [B4, B5]]]),
        (C1, B4, [(B1, B4)], [], 
         [[[A0, A1, A2, A3], [B0, B1, B2, B3], [C1, B4, B5]],
          [[A0, A1, A2, A3], [B0, B1, B2, B3], [C1, B4, B5]]]),
    ]:
        tracks = ActionableTracks(deepcopy(graph))
        tracks.assign_tracklet_ids()
        action = ConnectTrackAction(
            node_id1=node_id1,
            node_id2=node_id2,
            reconnect_others=reconnect_others,
        )
        action.apply(tracks)

        target_graph = deepcopy(graph)
        for edge in remove_edges:
            target_graph.remove_edge(*edge)
        target_graph.add_edge(node_id1, node_id2, {})
        if reconnect_others:
            for edge in add_edges_reconnect:
                target_graph.add_edge(*edge, {})
                
        # Check if isomorphic
        _assert_graphs_isomorphic(tracks.graph, target_graph)
        # Check tracklet IDs
        _compare_tracklet_id_assignments(
            tracklet_node_sets[1 if reconnect_others else 0],
            tracks.graph,
        )        

def test_connect_track_action_validates_time_ordering():
    graph = _make_empty_graph()
    late = _add_node(graph, t=5, offset=0)
    early = _add_node(graph, t=3, offset=10)
    tracks = ActionableTracks(graph)

    action = ConnectTrackAction(
        node_id1=late,
        node_id2=early,
        reconnect_others=False,
    )

    with pytest.raises(ValueError):
        action.apply(tracks)


def test_annotate_daughter_action_reuses_and_creates_nodes(sample_graph):
    graph, nodes_dict = sample_graph
    
    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]

    parent = B1
    existing = A2
    new_mask = _make_mask(offset=4)
    tracks = ActionableTracks(deepcopy(graph))
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
        [td.DEFAULT_ATTR_KEYS.T, tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert created_attrs[td.DEFAULT_ATTR_KEYS.T][0] == 2
    assert created_attrs[tracks.mask_attr_name][0] == new_mask
    assert np.array_equal(
        np.array(created_attrs[tracks.bbox_attr_name][0]), new_mask.bbox
    )

    _compare_tracklet_id_assignments(
        [[A0, A1], [A2, A3], [B0, B1], [B2, B3], [B4, B5], [new_node_id], [C1]],
        tracks.graph,
    )

    target_graph = deepcopy(graph)
    target_graph.add_edge(parent, existing, {})
    target_graph.remove_edge(A1, A2)
    new_node_id_target = target_graph.add_node(
        {
            td.DEFAULT_ATTR_KEYS.T: 2,
            tracks.mask_attr_name: new_mask,
            tracks.bbox_attr_name: new_mask.bbox,
            tracks.tracklet_id_attr_name: -1,
            "termination_annotation": "",
        }
    )
    target_graph.add_edge(parent, new_node_id_target, {})
    target_graph.remove_edge(B1, B2)
    target_graph.remove_edge(B1, B4)
    _assert_graphs_isomorphic(tracks.graph, target_graph)


def test_merge_labels_action_unifies_masks_and_removes_source_node(sample_graph):
    graph, nodes_dict = sample_graph

    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]
    
    tracks = ActionableTracks(deepcopy(graph))
    masks = graph.filter(node_ids=[A2, B2, B4]).node_attrs(
        [tracks.mask_attr_name]
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
        [tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    expected_mask = A2_mask | B2_mask
    assert filtered[tracks.mask_attr_name][0] == expected_mask
    assert np.array_equal(
        np.array(filtered[tracks.bbox_attr_name][0]), expected_mask.bbox
    )
    _compare_tracklet_id_assignments(
        [[A0, A1], [A3], [B0, B1], [B2, B3], [B4, B5], [C1]],
        tracks.graph,
    )


@pytest.mark.parametrize("delete_successor_tracklet", [False, True])
def test_annotate_termination_action_updates_annotation_and_handles_successors(
    sample_graph, delete_successor_tracklet: bool
):
    graph, nodes_dict = sample_graph
    A0, A1, A2, A3 = nodes_dict["A"]
    B0, B1, B2, B3, B4, B5 = nodes_dict["B"]
    C1, = nodes_dict["C"]

    for node_id, expected_remaining_node_ids, expected_groups in [
        (
            B1,
            { True: {A0, A1, A2, A3, B0, B1, B2, B3, B4, B5, C1},
              False: {A0, A1, A2, A3, B0, B1, B2, B3, B4, B5, C1},
            }[delete_successor_tracklet],
            { 
             True:  [[A0, A1, A2, A3],[B0, B1],[B2, B3],[B4, B5],[C1]],
             False: [[A0, A1, A2, A3],[B0, B1],[B2, B3],[B4, B5],[C1]],
            }[delete_successor_tracklet],
        ),
        (
            B2,
            { True: {A0, A1, A2, A3, B0, B1, B2, B4, B5, C1},
              False: {A0, A1, A2, A3, B0, B1, B2, B3, B4, B5, C1},
            }[delete_successor_tracklet],
            { 
             True:  [[A0, A1, A2, A3],[B0, B1],[B2],[B4, B5],[C1]],
             False: [[A0, A1, A2, A3],[B0, B1],[B2], [B3],[B4, B5],[C1]],
            }[delete_successor_tracklet],
        ),
    ]:
        tracks = ActionableTracks(deepcopy(graph))
        tracks.assign_tracklet_ids()

        action = AnnotateTerminationAction(
            node_id=node_id,
            termination_annotation="finished",
            delete_successor_tracklet=delete_successor_tracklet,
        )
        action.apply(tracks)

        attrs = tracks.graph.filter(node_ids=[node_id]).node_attrs(
            [tracks.termination_annotation_attr_name]
        )
        assert attrs[tracks.termination_annotation_attr_name][0] == "finished"
        assert tracks.graph.successors(node_id).is_empty()

        remaining_node_ids = set(tracks.graph.node_ids())
        assert remaining_node_ids == expected_remaining_node_ids

        _compare_tracklet_id_assignments(expected_groups, tracks.graph)

