import abc
from . import actionable_tracks, utils
from dataclasses import dataclass
import tracksdata as td
import polars as pl

ActionableTracks = actionable_tracks.ActionableTracks

@dataclass
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

def update_tracklet_ids(tracks: ActionableTracks, node_ids: list[int]):
    """Update tracklet IDs for the specified node IDs.

    Parameters
    ----------
    tracks : ActionableTracks
        The ActionableTracks instance managing the graph.
    node_ids : list[int]
        List of node IDs to update tracklet IDs for.
    """
    tree = tracks.graph.assign_tracklet_ids(
        output_key=tracks.tracklet_id_attr_name,
        node_ids=node_ids,
    )
    tracks.update_safe_tracklet_id(
        max(tree.nodes())
    )

class RedrawMaskAction(Action):
    node_id: int
    new_mask: td.nodes.Mask

    def apply(self, tracks: ActionableTracks):
        filtered = tracks.graph.filter(node_ids=[self.node_id])
        self.old_mask = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[tracks.mask_attr_name].first()
        tracks.graph.update_node_attrs(
            attrs={tracks.mask_attr_name: [self.new_mask],
                   tracks.bbox_attr_name: [self.new_mask.bbox]},
            node_ids=[self.node_id]
        )
        
class ConnectTrackAction(Action):
    node_id1: int
    node_id2: int 
    reconnect_others: bool = False

    def apply(self, tracks: ActionableTracks):
        times = utils.get_times(
            tracks.graph, [self.node_id1, self.node_id2]
    )

        if times[self.node_id1] >= times[self.node_id2]:
            raise ValueError("node_id1 must have an earlier time than node_id2 to connect.")

        successor_node_ids = utils.remove_successor_edges(tracks.graph, self.node_id1)
        predecessor_node_ids = utils.remove_predecessor_edges(tracks.graph, self.node_id2)
        tracks.graph.add_edge(self.node_id1, self.node_id2, {})
        if self.reconnect_others:
            for succ_id in successor_node_ids:
                tracks.graph.add_edge(self.node_id2, succ_id, {})
            for pred_id in predecessor_node_ids:
                tracks.graph.add_edge(pred_id, self.node_id1, {})
        
        update_tracklet_ids(
            tracks, 
            node_ids=[self.node_id1, self.node_id2, *successor_node_ids, *predecessor_node_ids]
        )
        # TODO set undo action
       
class AnnotateDaughterAction(Action):
    node_id: int
    daughters: list[int|tuple[int, td.nodes.Mask]]

    def apply(self, tracks: ActionableTracks):
        successor_node_ids = utils.remove_successor_edges(tracks.graph, self.node_id)
        daughter_node_ids = []
        for daughter in self.daughters:
            if isinstance(daughter, int):
                daughter_node_ids.append(daughter)
                tracks.graph.add_edge(self.node_id, daughter, {})
            elif isinstance(daughter, tuple):
                frame, mask = daughter
                new_node_id = tracks.graph.add_node(
                    attrs={
                        td.DEFAULT_ATTR_KEYS.MASK: mask,
                        td.DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
                        td.DEFAULT_ATTR_KEYS.T: frame,
                    }
                )
                daughter_node_ids.append(new_node_id)
                tracks.graph.add_edge(self.node_id, new_node_id, {})
        update_tracklet_ids(
            tracks, 
            node_ids=[self.node_id, *daughter_node_ids, *successor_node_ids]
        ) 
        
        
        
class MergeLabelsAction(Action):
    node_id_target: int
    node_id_merged: int
    
    def apply(self, tracks: ActionableTracks):
        # Get masks from both nodes
        filtered = tracks.graph.filter(node_ids=[self.node_id_target, self.node_id_merged])
        mask1 = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[tracks.mask_attr_name].to_list()[0]
        mask2 = filtered.node_attrs(attr_keys=[tracks.mask_attr_name])[tracks.mask_attr_name].to_list()[1]
        merged_mask = mask1 | mask2  # Union of the two masks
        tracks.graph.update_node_attrs(
            attrs={tracks.mask_attr_name: [merged_mask],
                   tracks.bbox_attr_name: [merged_mask.bbox]},
            node_ids=[self.node_id_target]
        )
        tracks.graph.remove_node(self.node_id_merged)
        update_tracklet_ids(
            tracks, 
            node_ids=[self.node_id_target]
        )
       
class AnnotateTerminationAction(Action):
    node_id: int
    termination_annotation: str = "terminated"
    delete_successors: bool = False

    def apply(self, tracks: ActionableTracks):
        utils.remove_successor_edges(tracks.graph, self.node_id)
        tracks.graph.update_node_attrs(
            attrs={tracks.termination_annotation_attr_name: [self.termination_annotation]},
            node_ids=[self.node_id]
        )
        data = tracks.graph.filter(node_ids=[self.node_id])\
            .node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.T, tracks.tracklet_id_attr_name])\
            .to_dicts()[0]
        time = data[tracks.time_attr_name]
        tracklet_id = data[tracks.tracklet_id_attr_name]
        query = (
            (pl.col(tracks.time_attr_name) > time) &
            (pl.col(tracks.tracklet_id_attr_name) == tracklet_id)
        )
        successor_node_id_df = tracks.graph.filter(query)\
            .node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID])
        if self.delete_successors and len(successor_node_id_df) > 0:
            for node_id in successor_node_id_df[td.DEFAULT_ATTR_KEYS.NODE_ID]:
                tracks.graph.remove_node(node_id)
    
