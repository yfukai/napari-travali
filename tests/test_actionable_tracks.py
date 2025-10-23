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
    update_tracklet_ids,
)
from napari_travali2.actionable_tracks.actionable_tracks import ActionableTracks


def _make_graph() -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
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


def test_update_tracklet_ids_assigns_consistent_ids():
    graph = _make_graph()
    node_ids = [
        _add_node(graph, t=frame, offset=frame * 10) for frame in range(3)
    ]
    graph.add_edge(node_ids[0], node_ids[1], {})
    graph.add_edge(node_ids[1], node_ids[2], {})
    tracks = ActionableTracks(graph)

    update_tracklet_ids(tracks, node_ids)

    df = graph.filter(node_ids=node_ids).node_attrs(
        [td.DEFAULT_ATTR_KEYS.TRACKLET_ID]
    )
    assigned_ids = df[td.DEFAULT_ATTR_KEYS.TRACKLET_ID].to_list()
    assert assigned_ids == [1, 1, 1]
    assert tracks.safe_tracklet_id >= 2


def test_redraw_mask_action_overwrites_mask_and_bbox():
    graph = _make_graph()
    node_id = _add_node(graph, t=0, offset=0)
    tracks = ActionableTracks(graph)

    filtered = graph.filter(node_ids=[node_id]).node_attrs(
        [tracks.mask_attr_name]
    )
    original_mask = filtered[tracks.mask_attr_name][0]

    action = RedrawMaskAction()
    action.node_id = node_id
    action.new_mask = _make_mask(offset=50, extra_pixel=True)
    action.apply(tracks)

    updated = graph.filter(node_ids=[node_id]).node_attrs(
        [tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert action.old_mask == original_mask
    assert updated[tracks.mask_attr_name][0] == action.new_mask
    assert np.array_equal(
        np.array(updated[tracks.bbox_attr_name][0]), action.new_mask.bbox
    )


def test_connect_track_action_links_nodes_and_reconnects_successors():
    graph = _make_graph()
    node1 = _add_node(graph, t=1, offset=0)
    node2 = _add_node(graph, t=4, offset=10)
    successor = _add_node(graph, t=6, offset=20)
    graph.add_edge(node1, successor, {})
    tracks = ActionableTracks(graph)

    action = ConnectTrackAction()
    action.node_id1 = node1
    action.node_id2 = node2
    action.reconnect_others = True
    action.apply(tracks)

    assert graph.has_edge(node1, node2)
    assert not graph.has_edge(node1, successor)
    assert graph.has_edge(node2, successor)

    involved_ids = [node1, node2, successor]
    df = graph.filter(node_ids=involved_ids).node_attrs(
        [td.DEFAULT_ATTR_KEYS.TRACKLET_ID]
    )
    assigned_ids = df[td.DEFAULT_ATTR_KEYS.TRACKLET_ID].to_list()
    assert all(tracklet_id >= 0 for tracklet_id in assigned_ids)


def test_connect_track_action_validates_time_ordering():
    graph = _make_graph()
    earlier = _add_node(graph, t=5, offset=0)
    later = _add_node(graph, t=3, offset=10)
    tracks = ActionableTracks(graph)

    action = ConnectTrackAction()
    action.node_id1 = earlier
    action.node_id2 = later

    with pytest.raises(ValueError):
        action.apply(tracks)


def test_annotate_daughter_action_reuses_and_creates_nodes():
    graph = _make_graph()
    parent = _add_node(graph, t=0, offset=0)
    existing = _add_node(graph, t=2, offset=5)
    graph.add_edge(parent, existing, {})
    tracks = ActionableTracks(graph)

    action = AnnotateDaughterAction()
    action.node_id = parent
    new_mask = _make_mask(offset=15)
    action.daughters = [existing, (4, new_mask)]
    before_ids = set(tracks.graph.node_ids())
    action.apply(tracks)
    after_ids = set(tracks.graph.node_ids())

    new_nodes = list(after_ids - before_ids)
    assert len(new_nodes) == 1
    new_node_id = new_nodes[0]

    assert graph.has_edge(parent, existing)
    assert graph.has_edge(parent, new_node_id)

    created_attrs = graph.filter(node_ids=[new_node_id]).node_attrs(
        [td.DEFAULT_ATTR_KEYS.T, tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    assert created_attrs[td.DEFAULT_ATTR_KEYS.T][0] == 4
    assert created_attrs[tracks.mask_attr_name][0] == new_mask
    assert np.array_equal(
        np.array(created_attrs[tracks.bbox_attr_name][0]), new_mask.bbox
    )


def test_merge_labels_action_unifies_masks_and_removes_source_node():
    graph = _make_graph()
    target_mask = _make_mask(offset=0)
    merged_mask = _make_mask(offset=2, extra_pixel=True)
    target = _add_node(graph, t=1, offset=0, mask=target_mask)
    merged = _add_node(graph, t=1, offset=2, mask=merged_mask)
    tracks = ActionableTracks(graph)

    action = MergeLabelsAction()
    action.node_id_target = target
    action.node_id_merged = merged
    action.apply(tracks)

    remaining_ids = list(graph.node_ids())
    assert merged not in remaining_ids
    filtered = graph.filter(node_ids=[target]).node_attrs(
        [tracks.mask_attr_name, tracks.bbox_attr_name]
    )
    expected_mask = target_mask | merged_mask
    assert filtered[tracks.mask_attr_name][0] == expected_mask
    assert np.array_equal(
        np.array(filtered[tracks.bbox_attr_name][0]), expected_mask.bbox
    )


@pytest.mark.parametrize(
    "delete_successors, expected_remaining", [(False, True), (True, False)]
)
def test_annotate_termination_action_updates_annotation_and_handles_successors(
    delete_successors: bool, expected_remaining: bool
):
    graph = _make_graph()
    parent = _add_node(graph, t=1, offset=0)
    child = _add_node(graph, t=3, offset=5)
    graph.add_edge(parent, child, {})
    graph.assign_tracklet_ids()
    tracks = ActionableTracks(graph)

    action = AnnotateTerminationAction()
    action.node_id = parent
    action.termination_annotation = "finished"
    action.delete_successors = delete_successors
    action.apply(tracks)

    attrs = graph.filter(node_ids=[parent]).node_attrs(
        [tracks.termination_annotation_attr_name]
    )
    assert attrs[tracks.termination_annotation_attr_name][0] == "finished"
    assert not graph.has_edge(parent, child)

    child_exists = child in set(graph.node_ids())
    assert child_exists is expected_remaining
