from __future__ import annotations

import numpy as np
import pytest
import tracksdata as td


@pytest.fixture
def make_empty_graph():
    def _make_empty_graph(graph_type) -> td.graph.BaseGraph:
        if graph_type == "inmem":
            graph = td.graph.RustWorkXGraph()
        elif graph_type == "sql":
            graph = td.graph.SQLGraph(drivername="sqlite", database=":memory:")
        for key, default in [
            (td.DEFAULT_ATTR_KEYS.MASK, None),
            (td.DEFAULT_ATTR_KEYS.BBOX, None),
            (td.DEFAULT_ATTR_KEYS.TRACKLET_ID, -1),
            ("termination_annotation", ""),
            ("verified", False),
        ]:
            graph.add_node_attr_key(key, default)
        return graph

    return _make_empty_graph


@pytest.fixture
def make_mask():
    def _make_mask(offset: int, extra_pixel: bool = False) -> td.nodes.Mask:
        mask_array = np.zeros((2, 2), dtype=bool)
        mask_array[0, 0] = True
        if extra_pixel:
            mask_array[1, 1] = True
        bbox = np.array(
            [offset, offset, offset + mask_array.shape[0], offset + mask_array.shape[1]]
        )
        return td.nodes.Mask(mask_array, bbox)

    return _make_mask


@pytest.fixture
def add_node(make_mask):
    def _add_node(
        graph: td.graph.BaseGraph,
        *,
        t: int,
        offset: int,
        mask: td.nodes.Mask | None = None,
    ) -> int:
        mask = mask or make_mask(offset)
        return graph.add_node(
            {
                td.DEFAULT_ATTR_KEYS.T: t,
                td.DEFAULT_ATTR_KEYS.MASK: mask,
                td.DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
                td.DEFAULT_ATTR_KEYS.TRACKLET_ID: -1,
                "termination_annotation": "",
                "verified": False,
            }
        )

    return _add_node


@pytest.fixture
def assert_graphs_isomorphic():
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

    return _assert_graphs_isomorphic


@pytest.fixture
def make_sample_graph(make_empty_graph, add_node):
    def _make_sample_graph(graph_type):
        """Create a sample tracks graph for testing."""

        graph = make_empty_graph(graph_type)

        # Build graph components
        # A chain
        A0 = add_node(graph, t=0, offset=0)
        A1 = add_node(graph, t=1, offset=1)
        A2 = add_node(graph, t=2, offset=2)
        A3 = add_node(graph, t=3, offset=3)
        graph.add_edge(A0, A1, {})
        graph.add_edge(A1, A2, {})
        graph.add_edge(A2, A3, {})

        # B branched with parent and children
        B0 = add_node(graph, t=0, offset=1)
        B1 = add_node(graph, t=1, offset=2)
        B2 = add_node(graph, t=2, offset=3)
        B4 = add_node(graph, t=2, offset=4)
        B3 = add_node(graph, t=3, offset=5)
        B5 = add_node(graph, t=3, offset=6)
        graph.add_edge(B0, B1, {})
        graph.add_edge(B1, B2, {})
        graph.add_edge(B1, B4, {})
        graph.add_edge(B2, B3, {})
        graph.add_edge(B4, B5, {})
        C0 = add_node(graph, t=1, offset=10)

        return graph, {
            "A": [A0, A1, A2, A3],
            "B": [B0, B1, B2, B3, B4, B5],
            "C": [C0],
        }

    return _make_sample_graph


@pytest.fixture
def compare_tracklet_id_assignments():
    def _compare_tracklet_id_assignments(
        expected_node_sets, graph_backend: td.graph.BaseGraph
    ):
        """Compare tracklet ID assignments in the graph backend to expected node sets. Copied from tracksdata tests."""

        ids_df = graph_backend.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.TRACKLET_ID]
        )
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

    return _compare_tracklet_id_assignments
