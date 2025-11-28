from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

import tracksdata as td

from . import utils
from .._logging import logger

if TYPE_CHECKING:
    from .actionable_tracks import ActionableTracks

class Action(abc.ABC):
    """Abstract base class for actions on actionable tracks."""
    
    @abc.abstractmethod
    def apply(self, tracks: "ActionableTracks"):
        """Apply the action to the tracks graph.

        Parameters
        ----------
        tracks : ActionableTracks
            The ActionableTracks instance managing the graph.
        """
        raise NotImplementedError("Subclasses must implement this method.")


def _get_mask_attrs(*, mask: td.nodes.Mask, 
                    node_attr_keys: list[str], 
                    tracks: ActionableTracks,  
                    frame: int | None = None,) -> dict[str, Any]:
    centroid_attr_keys = ["z","y","x"][-mask.mask.ndim:]

    attrs = {
        tracks.mask_attr_name: mask,
        tracks.bbox_attr_name: mask.bbox,
        tracks.time_attr_name: frame,
        tracks.termination_annotation_attr_name: "",
    }
    props = mask.regionprops()
    for key, val in zip(centroid_attr_keys, props.centroid):
        attrs[key] = val
    for node_attr_key in node_attr_keys:
        if node_attr_key not in attrs:
            val = getattr(props, node_attr_key, None)
            attrs[node_attr_key] = val
    return attrs

@dataclass
class AddNodeAction(Action):
    """Add a new node to the tracks graph."""

    frame: int
    tracklet_id: int
    mask: td.nodes.Mask
    connected_node_id: int | None = None

    def apply(self, tracks: ActionableTracks):
        """Add a new node with the specified frame and mask."""
        node_attr_keys = tracks.graph.node_attr_keys
        node_attr_keys.remove(tracks.tracklet_id_attr_name)
        node_attr_keys.remove(td.DEFAULT_ATTR_KEYS.NODE_ID)
        logger.debug(
            f"Preparing attributes for new node in frame {self.frame} with tracklet_id {self.tracklet_id}."
        )
        attrs = _get_mask_attrs(
            frame=self.frame,
            mask=self.mask,
            node_attr_keys=node_attr_keys,
            tracks=tracks,
        )
        attrs[tracks.tracklet_id_attr_name] = self.tracklet_id
        logger.info(f"Adding node with attributes {attrs}.")
        new_node_id = tracks.graph.add_node(attrs=attrs)
        logger.info(f"Added node {new_node_id} to graph.")
        if self.connected_node_id is not None:
            logger.debug(
                f"Connecting new node {new_node_id} to existing node {self.connected_node_id}."
            )
            times = utils.get_times(
                tracks.graph, [self.connected_node_id, new_node_id]
            )
            if times[self.connected_node_id] == times[new_node_id]:
                raise ValueError("Cannot connect nodes with the same time.")
            if times[self.connected_node_id] < times[new_node_id]:
                tracks.graph.add_edge(self.connected_node_id, new_node_id, {})
            else:
                tracks.graph.add_edge(new_node_id, self.connected_node_id, {})
            logger.info(
                f"Connected node {self.connected_node_id} with node {new_node_id}."
            )
        return new_node_id

@dataclass
class RedrawMaskAction(Action):
    """Replace the segmentation mask of a single node."""

    node_id: int
    new_mask: td.nodes.Mask

    def apply(self, tracks: ActionableTracks):
        """Overwrite the node mask and bounding box with the provided mask."""
        filtered = tracks.graph.filter(node_ids=[self.node_id])
        logger.debug(f"Fetched node {self.node_id} for redraw.")
        self.old_mask = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[
            tracks.mask_attr_name
        ].first()
        node_attr_keys = tracks.graph.node_attr_keys
        attrs = _get_mask_attrs(
            mask=self.new_mask,
            node_attr_keys=node_attr_keys,
            tracks=tracks,
        )
        attrs = {k: [v] for k, v in attrs.items() if v is not None}
        logger.info(
            f"Updating mask and bbox for node {self.node_id} with attrs {attrs}."
        )
        tracks.graph.update_node_attrs(
            attrs=attrs,
            node_ids=[self.node_id],
        )

@dataclass
class ConnectTrackAction(Action):
    """Connect two track nodes by inserting an edge and reassigning tracklet IDs."""

    node_id1: int
    node_id2: int
    reconnect_others: bool = False

    def apply(self, tracks: ActionableTracks):
        """Insert an edge from the earlier node to the later node and optionally reconnect neighbors."""
        times = utils.get_times(
            tracks.graph, [self.node_id1, self.node_id2]
        )
        logger.debug(
            f"Attempting to connect node {self.node_id1} (t={times[self.node_id1]}) -> node {self.node_id2} (t={times[self.node_id2]})."
        )

        if times[self.node_id1] >= times[self.node_id2]:
            raise ValueError("node_id1 must have an earlier time than node_id2 to connect.")

        successor_node_ids = utils.remove_successor_edges(
            tracks.graph, self.node_id1
        )
        logger.info(
            f"Removed successor edges for node {self.node_id1}: {successor_node_ids}."
        )
        predecessor_node_ids = utils.remove_predecessor_edges(
            tracks.graph, self.node_id2
        )
        logger.info(
            f"Removed predecessor edges for node {self.node_id2}: {predecessor_node_ids}."
        )
        tracks.graph.add_edge(self.node_id1, self.node_id2, {})
        logger.info(f"Added edge {self.node_id1} -> {self.node_id2}.")
        if self.reconnect_others:
            logger.debug("Reconnecting neighboring nodes.")
            if len(predecessor_node_ids) > 1:
                raise ValueError("Cannot reconnect predecessors when multiple exist.")
            for succ_id in successor_node_ids:
                for pred_id in predecessor_node_ids:
                    tracks.graph.add_edge(pred_id, succ_id, {})
                    logger.info(f"Added edge {pred_id} -> {succ_id}.")
        
        node_ids = [self.node_id1, self.node_id2, *successor_node_ids, *predecessor_node_ids]
        successor_node_ids2 = tracks.graph.successors(node_ids)
        node_ids.extend(sum(successor_node_ids2.values(), []))
        logger.info(f"Reassigning tracklet IDs for nodes: {node_ids}.")
        tracks.assign_tracklet_ids(
            list(set(node_ids)),
        )
        # TODO set undo action
       
@dataclass
class AnnotateDaughterAction(Action):
    """Create or connect daughter nodes from a parent division event."""

    node_id: int
    daughters: list[int|tuple[int, td.nodes.Mask]]

    def apply(self, tracks: ActionableTracks):
        """Attach provided daughter nodes or create new nodes at specified frames."""
        successor_node_ids = utils.remove_successor_edges(tracks.graph, self.node_id)
        daughter_node_ids = []
        relevant_node_ids = []
        logger.debug(f"Annotating daughters for node {self.node_id}.")
        logger.info(
            f"Removed successor edges for node {self.node_id}: {successor_node_ids}."
        )
        for daughter in self.daughters:
            if isinstance(daughter, int):
                relevant_node_ids.extend(
                    utils.remove_predecessor_edges(tracks.graph, daughter)
                )
                daughter_node_ids.append(daughter)
                logger.info(
                    f"Removed predecessor edges and will connect existing daughter node {daughter}."
                )
                tracks.graph.add_edge(self.node_id, daughter, {})
                logger.info(f"Added edge {self.node_id} -> {daughter}.")
            elif isinstance(daughter, tuple):
                frame, mask = daughter
                new_node_id = tracks.graph.add_node(
                    attrs={
                        tracks.mask_attr_name: mask,
                        tracks.bbox_attr_name: mask.bbox,
                        tracks.time_attr_name: frame,
                        tracks.tracklet_id_attr_name: -1,
                        tracks.termination_annotation_attr_name: "",
                        tracks.verified_attr_name: False,
                    }
                )
                daughter_node_ids.append(new_node_id)
                logger.info(f"Added daughter node {new_node_id} at frame {frame}.")
                tracks.graph.add_edge(self.node_id, new_node_id, {})
                logger.info(f"Added edge {self.node_id} -> {new_node_id}.")
        tracks.assign_tracklet_ids(
            [self.node_id, *daughter_node_ids, *successor_node_ids],
        )
        logger.info(
            f"Reassigned tracklet IDs for node {self.node_id} and daughters {daughter_node_ids}."
        )
        return daughter_node_ids


@dataclass
class MergeLabelsAction(Action):
    """Merge the segmentation labels of two nodes into the target node."""

    node_id_target: int
    node_id_merged: int

    def apply(self, tracks: ActionableTracks):
        """Fuse masks from two nodes, remove the merged node, and refresh tracklet IDs."""
        # Get masks from both nodes
        filtered = tracks.graph.filter(node_ids=[self.node_id_target, self.node_id_merged])
        logger.debug(
            f"Fetched masks for merging nodes {self.node_id_target} and {self.node_id_merged}."
        )
        mask1 = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[tracks.mask_attr_name].to_list()[0]
        mask2 = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[tracks.mask_attr_name].to_list()[1]
        merged_mask = mask1 | mask2  # Union of the two masks
        logger.info(f"Updating merged mask on node {self.node_id_target}.")
        tracks.graph.update_node_attrs(
            attrs={
                tracks.mask_attr_name: [merged_mask],
                tracks.bbox_attr_name: [merged_mask.bbox],
            },
            node_ids=[self.node_id_target],
        )
        successor_node_ids = utils.remove_successor_edges(
            tracks.graph, self.node_id_merged
        )
        predecessor_node_ids = utils.remove_predecessor_edges(
            tracks.graph, self.node_id_merged
        )
        logger.info(
            f"Removed edges around merged node {self.node_id_merged} (successors: {successor_node_ids}, predecessors: {predecessor_node_ids})."
        )
        tracks.graph.remove_node(self.node_id_merged)
        logger.info(f"Removed merged node {self.node_id_merged}.")
        tracks.assign_tracklet_ids(
            [self.node_id_target, *successor_node_ids, *predecessor_node_ids],
        )
        logger.info(
            f"Reassigned tracklet IDs after merge with target {self.node_id_target}."
        )

@dataclass
class AnnotateTerminationAction(Action):
    """Mark a node as terminated and optionally prune downstream nodes."""

    node_id: int
    termination_annotation: str = "terminated"
    delete_successor_tracklet: bool = False

    def apply(self, tracks: ActionableTracks):
        """Set a termination annotation and optionally delete subsequent nodes in the tracklet."""
        successor_node_ids = utils.remove_successor_edges(tracks.graph, self.node_id)
        logger.info(
            f"Removed successor edges for node {self.node_id}: {successor_node_ids}."
        )
        tracks.graph.update_node_attrs(
            attrs={tracks.termination_annotation_attr_name: [self.termination_annotation]},
            node_ids=[self.node_id]
        )
        logger.info(
            f"Annotated node {self.node_id} with termination '{self.termination_annotation}'."
        )
        if self.delete_successor_tracklet:
            time, tracklet_id = tracks.graph.filter(
                node_ids=[self.node_id]
            ).node_attrs(
                attr_keys=[tracks.time_attr_name, tracks.tracklet_id_attr_name]
            ).to_dicts()[0].values()
            logger.debug(
                f"Deleting successor tracklet for node {self.node_id} at time {time} with tracklet {tracklet_id}."
            )
            deleted_node_ids = tracks.graph.filter(
                td.NodeAttr(tracks.time_attr_name) > time,
                td.NodeAttr(tracks.tracklet_id_attr_name) == tracklet_id,
            ).node_ids()
            for del_node_id in deleted_node_ids:
                tracks.graph.remove_node(del_node_id)
            logger.info(
                f"Removed nodes {deleted_node_ids} from successor tracklet of node {self.node_id}."
            )
            successor_node_ids = [*successor_node_ids, *deleted_node_ids]
        tracks.assign_tracklet_ids([self.node_id, *successor_node_ids])
        logger.info(
            f"Reassigned tracklet IDs for node {self.node_id} and successors {successor_node_ids}."
        )
