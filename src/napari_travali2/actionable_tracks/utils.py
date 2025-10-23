import polars as pl
import tracksdata as td
import numpy as np

def _remove_connected_edges(graph, node_id, direction: str):
    if direction == 'predecessors':
        neighbors : pl.DataFrame = graph.predecessors(node_id)
    elif direction == 'successors':
        neighbors : pl.DataFrame = graph.successors(node_id)
    neighbor_node_ids = neighbors[td.DEFAULT_ATTR_KEYS.NODE_ID] if len(neighbors) > 0 else []
    for neighbor_node_id in neighbor_node_ids:
        graph.remove_edge(node_id, neighbor_node_id)
    return neighbor_node_ids

def remove_predecessor_edges(graph, node_id):
    return _remove_connected_edges(graph, node_id, direction='predecessors')
def remove_successor_edges(graph, node_id):
    return _remove_connected_edges(graph, node_id, direction='successors')

def get_times(graph: td.graph.BaseGraph, node_ids: list[int]) -> dict[int, int]:
    """Get the time attributes for the specified node IDs.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The tracks graph.
    node_ids : list[int]
        List of node IDs to retrieve time attributes for.

    Returns
    -------
    dict[int, int]
        A dictionary mapping node IDs to their corresponding time attributes.
    
    """
    row_dicts = graph.filter(node_ids=node_ids)\
       .node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.T])\
       .to_dicts()
    return {
        int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]): int(row[td.DEFAULT_ATTR_KEYS.T]) 
    for row in row_dicts}

    
