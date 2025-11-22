import os
os.environ["NAPARI_ASYNC"] = "1"
from laptrack import datasets, LapTrack
import napari
import pandas as pd
import tracksdata as td
from napari import Viewer
import napari_travali2 as travali2
from napari_travali2._logging import logger
from napari_travali2.actionable_tracks import actionable_tracks as at

image, labels = datasets.cell_segmentation()

viewer = napari.Viewer()
viewer.add_image(image, name='image')
viewer.add_labels(labels, name='labels')

node_op = td.nodes.RegionPropsNodes(
    extra_properties=["area","intensity_mean"]
)
graph = td.graph.SQLGraph(drivername="sqlite", database="test.db")
#graph = td.graph.RustWorkXGraph()
node_op.add_nodes(graph, labels=labels, intensity_image=image)

nodes_df = graph.node_attrs().to_pandas()
# Expand centroid tuple into separate columns

lt = LapTrack(
    metric="sqeuclidean",
    cutoff=15**2,
    splitting_metric="sqeuclidean",
    splitting_cutoff=15**2
)
track_df, split_df, _ = lt.predict_dataframe(nodes_df, 
                                             frame_col='t', 
                                             coordinate_cols=['y', 'x'],
                                             only_coordinate_cols=False,
                                             index_offset=1)
track_df2 = track_df.reset_index().sort_values("frame")
# Updating the graph with track IDs
if "track_id" not in graph.node_attr_keys:
    graph.add_node_attr_key("track_id", 0)
graph.update_node_attrs(
    attrs = {"track_id":track_df["track_id"].to_list()},
    node_ids=track_df["node_id"].to_list(),
)
for _, grp in track_df2.groupby("track_id"):
    node_ids = grp["node_id"].to_list()
    graph.bulk_add_edges([
        {"source_id":node_ids[i], "target_id":node_ids[i+1]} for i in range(len(node_ids)-1)
    ])

# Adding splitting edges
first_node_df = track_df2.drop_duplicates(subset=["track_id"], keep="first")
last_node_df = track_df2.drop_duplicates(subset=["track_id"], keep="last")
split_df2 = split_df.merge(
    first_node_df[["track_id", "node_id"]].rename(columns={"node_id":"child_node_id"}),
    left_on="child_track_id",
    right_on="track_id",
    how="left"
).merge(
    last_node_df[["track_id", "node_id"]].rename(columns={"node_id":"parent_node_id"}),
    left_on="parent_track_id",
    right_on="track_id",
    how="left"
)[["parent_node_id", "child_node_id"]]
graph.bulk_add_edges(
    [{"source_id":row["parent_node_id"], "target_id":row["child_node_id"]} for _, row in split_df2.iterrows()]
)

tracks = at.ActionableTracks(graph, tracklet_id_attr_name="track_id")

logger.setLevel("DEBUG")
logger.info("Starting napari-travali2")

viewer = Viewer()

widget = travali2.StateMachineWidget(viewer, tracks, image, 
                            verified_track_ids=[],
                            candidate_track_ids=[],
                            crop_size=2048,
                            tracklet_id_attr_name="track_id")
viewer.window.add_dock_widget(widget, area="right")
viewer.dims.set_current_step(0,0)

napari.run()