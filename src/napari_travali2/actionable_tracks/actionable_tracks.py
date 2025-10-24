import tracksdata as td
from .action import Action
import polars as pl
   

class ActionableTracks:
    time_attr_name: str = td.DEFAULT_ATTR_KEYS.T
    mask_attr_name: str = td.DEFAULT_ATTR_KEYS.MASK
    bbox_attr_name: str = td.DEFAULT_ATTR_KEYS.BBOX
    tracklet_id_attr_name: str = td.DEFAULT_ATTR_KEYS.TRACKLET_ID
    termination_annotation_attr_name: str = "termination_annotation"

    def __init__(self, 
                 graph: td.graph.BaseGraph, 
                 safe_tracklet_id : int | None = None) -> None:
        if self.tracklet_id_attr_name not in graph.node_attr_keys:
            raise ValueError(f"tracklet_id_column '{self.tracklet_id_attr_name}' not found in graph node attributes.")
        self.graph :td.graph.BaseGraph = graph

        self.action_history = []
        if safe_tracklet_id is None:
            self.initialize_safe_tracklet_id()
        else:
            self._safe_tracklet_id = safe_tracklet_id


    def initialize_safe_tracklet_id(self) -> int:
        """Initialize the safe label counter based on existing track IDs.
        
        Returns
        -------
        int
            The assigned safe tracklet ID to use.
            
        """
        df = self.graph.node_attrs(
            attr_keys=self.tracklet_id_attr_name
        ).filter(pl.col(self.tracklet_id_attr_name) != -1)
        if len(df) == 0:
            self._safe_tracklet_id = 1
        else:
            self._safe_tracklet_id = int(df[self.tracklet_id_attr_name].max() + 1)
        return self._safe_tracklet_id
    
    def _update_safe_tracklet_id(self, new_tracklet_id=0) -> int:
        """Update the safe label counter to ensure a unique track ID.

        Parameters
        ----------
        new_label : int
            The new label to consider when updating the safe label counter.
            
        Returns
        -------
        int
            The updated safe tracklet ID.
            
        """
        self._safe_tracklet_id = max(self._safe_tracklet_id, new_tracklet_id + 1)
        return self._safe_tracklet_id

    def assign_tracklet_ids(self, node_ids: list[int] | None = None):
        """Update tracklet IDs for specified nodes and their connected components.

        Parameters
        ----------
        node_ids : list[int]
            List of node IDs to update tracklet IDs for.
        """
        tree = self.graph.assign_tracklet_ids(
            output_key=self.tracklet_id_attr_name,
            node_ids=node_ids,
            tracklet_id_offset=self.safe_tracklet_id,
        )
        self._update_safe_tracklet_id(max(tree.nodes()))

    @property
    def safe_tracklet_id(self):
        """Get a safe (unused) track ID.

        Returns
        -------
        int
            A track ID that is safe to use for new tracks.
        """
        return self._safe_tracklet_id

    def apply(self, action: "Action"):
        """Apply an action to the tracks graph.

        Parameters
        ----------
        action : ActionableTrackAction
            The action to apply.
        """
        action.apply(self)
        self.action_history.append(action)



