from ._logging import logger, log_error
import tracksdata as td
import polars as pl
import numpy as np

@log_error
def find_track_by_coordinates(
        bbox_spatial_filter, 
        data_coordinates: np.ndarray) -> int | None:
    logger.debug(f"data_coordinates: {data_coordinates}")
    nodes_df = bbox_spatial_filter[tuple([
        slice(c,c) for c in data_coordinates
    ])].node_attrs(attr_keys=["node_id", "label"])
    logger.debug("node_df built.")
    if nodes_df.is_empty():
        logger.info("No nodes in the bbox.")
        return None
    track_id = nodes_df["label"].to_list()[0]
    return track_id

@log_error
def find_track_successors(graph, track_id, track_id_attr_key) -> tuple[int, pl.DataFrame]:
    # TODO make the following faster
    df = graph.filter(td.NodeAttr(track_id_attr_key) == track_id).node_attrs(attr_keys=["node_id","t"]).sort("t")
    sorted_node_ids = df["node_id"]
    logger.debug(f"{df=}")
    last_node_id = sorted_node_ids.last()
    logger.debug(f"Sorted_node_ids built. Last node id: {last_node_id}. track_id: {track_id}, track_id_attr_key: {track_id_attr_key}")
    #successors_df = graph.successors(last_node_id, return_attrs=True, attr_keys=["track_id"])
    successors_df = graph.successors(last_node_id, attr_keys=[track_id_attr_key], return_attrs=True)
    logger.debug("successors_df built.")
    return successors_df

