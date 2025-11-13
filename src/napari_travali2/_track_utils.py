from ._logging import logger, log_error
import tracksdata as td
import polars as pl

@log_error
def find_track_successors(graph, bbox_spatial_filter, data_coordinates) -> tuple[int, pl.DataFrame]:
    logger.debug(f"data_coordinates: {data_coordinates}")
    nodes_df = bbox_spatial_filter[tuple([
        slice(c,c) for c in data_coordinates
    ])].node_attrs(attr_keys=["node_id", "label"])
    logger.debug("node_df built.")
    if nodes_df.is_empty():
        logger.info("No nodes in the bbox.")
        return -1, pl.DataFrame()
    track_id = nodes_df["label"].to_list()[0]
    # TODO make the following faster
    sorted_node_ids = graph.filter(td.NodeAttr("label") == track_id).node_attrs(attr_keys=["node_id","t"]).sort("t")["node_id"]
    logger.debug("sorted_node_ids built.")
    last_node_id = sorted_node_ids.last()
    #successors_df = graph.successors(last_node_id, return_attrs=True, attr_keys=["track_id"])
    successors_df = graph.successors(last_node_id, attr_keys=["label"])
    logger.debug("successors_df built.")
    return track_id, successors_df

