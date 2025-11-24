from transitions import Machine
from magicgui.widgets import Container, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS, STATE_EXPLANATION
from ._logging import logger, log_error
from ._gui_utils import choose_direction_by_mbox, get_annotation_of_track_end,choose_division_by_mbox, ask_draw_label
from ._consts import SELECTED_COLOR, DAUGHTER_COLORS
from ._track_utils import find_track_successors, find_track_by_coordinates
import numpy as np
from dask import array as da
import polars as pl
from copy import deepcopy
from .actionable_tracks import actionable_tracks as at, action
from napari.utils.colormaps import label_colormap
import tracksdata as td
from dataclasses import dataclass

def default_colormap():
    return label_colormap(
            49, 0, background_value=0
    )

SHOW_SELECTED_LABEL_STATES = [
    ViewerState.LABEL_SELECTED,
    ViewerState.LABEL_REDRAW,
    ViewerState.DAUGHTER_CHOOSE_MODE,
    ViewerState.DAUGHTER_DRAW,
]

LAYER_NAMES = [
    "image",
    "labels",
    "verified",
    "cropped_image",
    "cropped_labels",
    "cropped_verified",
    "redraw",
]
@dataclass
class _LayerConfig:
    visible_layers : list[str]
    active_layer : str 
select_config = _LayerConfig(["cropped_image", "cropped_labels", "cropped_verified"], "cropped_labels")
redraw_config = _LayerConfig(["cropped_image", "redraw"],"redraw")

LAYER_CONFIGS = { 
    ViewerState.SELECT_REGION: _LayerConfig(["image", "labels","verified"], "labels"),
    ViewerState.ALL_LABEL: select_config,
    ViewerState.LABEL_SELECTED: select_config,
    ViewerState.LABEL_REDRAW: redraw_config,
    ViewerState.LABEL_SWITCH: select_config,
    ViewerState.DAUGHTER_SWITCH: select_config, 
    ViewerState.DAUGHTER_DRAW: redraw_config,
    ViewerState.DAUGHTER_CHOOSE_MODE: select_config,
}

@dataclass
class _SelectedTrackInfo:
    graph: td.graph.BaseGraph
    tracklet_id_attr_name: str
    track_id: int
    def __post_init__(self):
        self.refresh()
   
    def refresh(self):
        self.subgraph: td.graph.BaseGraph = self.graph.filter(
            td.NodeAttr(self.tracklet_id_attr_name) == self.track_id
        ).subgraph()
        self.frames: list[int] = self.subgraph.node_attrs(attr_keys=["t"])["t"].to_list()
        self.update_daughters()
 
    def update_daughters(self):
        logger.info(f"Updating daughters for track ID: {self.track_id}")
        successors_df = find_track_successors(self.graph, self.track_id,
                                              self.tracklet_id_attr_name)
        if len(successors_df) == 0:
            logger.info("No successors found.")
            successor_track_ids = []
        else:
            successor_track_ids = successors_df[self.tracklet_id_attr_name].to_list()
        logger.info(f"Successor track IDs: {successor_track_ids}")
        self.daughter_track_ids = successor_track_ids

class StateMachineWidget(Container):

    def __initialize_layers(self):
        if "Cropped Image" in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers["Cropped Image"])
        self._cropped_image_layer = self._viewer.add_image(
            da.zeros(self.cropped_shape, dtype=self.image_dtype), 
            name="Cropped Image")
        
        if "Cropped Labels" in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers["Cropped Labels"])
        self._cropped_labels_layer = self._viewer.add_labels(
            da.zeros(self.cropped_shape, dtype=self.label_dtype), 
            name="Cropped Labels")

        if "Cropped Verified" in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers["Cropped Verified"])
        self._cropped_verified_layer = self._viewer.add_labels(
            np.zeros(self.cropped_shape, dtype=bool), 
            name="Cropped Verified", cache=False)
        self._cropped_verified_layer.colormap = {0: (0,0,0,0), None:(1,0,0,1)}
        self._cropped_verified_layer.contour = 4

        if "Redraw" in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers["Redraw"])
        self._redraw_layer = self._viewer.add_labels(
            np.zeros(self.space_like_shape, dtype=bool), 
            name="Redraw", cache=False)


    def __refresh_labels(self):
        self._track_gav._cache._store.clear() # Should be removed after fixing tracksdata cache update issue
        self._verified_gav._cache._store.clear() # Should be removed after fixing tracksdata cache update issue
        self._labels_layer.refresh()
        self._cropped_labels_layer.refresh()

    def __bind_events(self):
        # XXX possibly refactor this
        self._viewer.bind_key("o", lambda event: self.o_typed(), overwrite=True)
        self._viewer.bind_key("r", lambda event: self.r_typed(), overwrite=True)
        self._viewer.bind_key("s", lambda event: self.s_typed(), overwrite=True)
        self._viewer.bind_key("d", lambda event: self.d_typed(), overwrite=True)
        self._viewer.bind_key("t", lambda event: self.t_typed(), overwrite=True)
        self._viewer.bind_key("n", lambda event: self.n_typed(), overwrite=True)
        self._viewer.bind_key("c", lambda event: self.c_typed(), overwrite=True)

    def __bind_mouse_events(self):
        self._viewer.mouse_drag_callbacks.append(self._track_clicked_wrapper)        
        self._labels_layer.mouse_drag_callbacks.append(self._region_clicked_wrapper)

    def __unbind_mouse_events(self):
        self._viewer.mouse_drag_callbacks.remove(self._track_clicked_wrapper)        
        self._labels_layer.mouse_drag_callbacks.remove(self._region_clicked_wrapper)

    def __init__(self, 
                 viewer: Viewer, 
                 tracks: at.ActionableTracks,
                 image: np.ndarray,
                 crop_size=1024,
                 tracklet_id_attr_name="label",
                 ):
        super().__init__()

        self._viewer = viewer
        self._tracks = tracks
        self._image = image
        self._track_gav = td.array.GraphArrayView(
            tracks.graph, 
            shape=tuple(image.shape), 
            attr_key=tracklet_id_attr_name
        )
        self._verified_gav = td.array.GraphArrayView(
            tracks.graph, 
            shape=tuple(image.shape), 
            attr_key="verified",
            dtype=np.uint8,
        )
        self._image_layer = viewer.add_image([image, image[::2,::2]], name="Image")
        self._labels_layer = viewer.add_labels([self._track_gav, self._track_gav[::2,::2]], name="Labels")
        self._verified_layer = viewer.add_labels([self._verified_gav, self._verified_gav[::2,::2]], name="Verified", cache=False)
        self._verified_layer.colormap = {0: (0,0,0,0), None:(1,0,0,1)}
        self._verified_layer.contour = 4

        self.crop_size = crop_size
        self.tracklet_id_attr_name = tracklet_id_attr_name
        self._selected_track: _SelectedTrackInfo|None = None

        self.time_like_shape = self._track_gav.shape[:-2]
        self.space_like_shape = self._track_gav.shape[-2:]
        self.image_dtype = image.dtype.name
        self.label_dtype = self._track_gav.dtype
        self.cropped_shape = (*self.time_like_shape, crop_size, crop_size)

        logger.info("StateMachineWidget initialized.")
        logger.info(f"Image dtype: {self.image_dtype}, Label dtype: {self.label_dtype}")
        logger.info(f"Image shape: {self._image.shape}, GAV shape: {self._track_gav.shape}")
        logger.info(f"Cropped shape: {self.cropped_shape}")
        logger.info(f"Time-like shape: {self.time_like_shape}, Space-like shape: {self.space_like_shape}")

        self.__initialize_layers()
        # Add a label to show the current state
        self._state_label = Label(value=f"Before initialization.")
        #self._state_label.setText(f"Current State: {ViewerState.ALL_LABEL}")
        self.extend([
            self._state_label
        ])
        
        self.machine = Machine(model=self, 
                               states=ViewerState, 
                               initial=ViewerState.SELECT_REGION,
                               transitions=TRANSITIONS,
                               after_state_change="update_viewer_status",
                               ignore_invalid_triggers=True)
        
        self.__bind_events()
        self.__bind_mouse_events()

        #self.update_finalized_point_layer()
        self.update_viewer_status()  
        
    
    ############### Viewer status update ###############
       
    @log_error    
    def update_viewer_status(self,*_args):
        self.__refresh_labels()
        self._state_label.value = ("========================\n"
                                  f"Current State: {self.state.name}\n"
                                  "========================\n"
                                  "\n"
                                  f"{STATE_EXPLANATION[self.state]}")
        cfg = LAYER_CONFIGS[self.state]
        for layer_name in LAYER_NAMES:
            layer = getattr(self, f"_{layer_name}_layer")
            layer.visible = layer_name in cfg.visible_layers
            if layer.visible:
                logger.info(f"Layer '{layer_name}' set to visible.")
            else:
                logger.info(f"Layer '{layer_name}' set to invisible.")
            if layer_name == cfg.active_layer:
                self._viewer.layers.selection.active = layer

            if self.state not in SHOW_SELECTED_LABEL_STATES:
                self.set_default_colormap()
            else:
                self.set_selected_colormap()

        if cfg.active_layer == "redraw":
            self._redraw_layer.selected_label = 1
            self._redraw_layer.mode = "paint"
            self.__unbind_mouse_events()
        else:
            self.__bind_mouse_events()
            # XXX better if I can set the viewer.dims not to change
    

    @log_error
    def set_default_colormap(self):
        self._cropped_labels_layer.colormap = default_colormap()
    
    @log_error
    def set_selected_colormap(self):
        assert self._selected_track is not None, "No selected track."
        self._cropped_labels_layer.colormap = {0:(0,0,0,0), 
                                              self._selected_track.track_id:SELECTED_COLOR,
                                              **{d:c for d,c in zip(self._selected_track.daughter_track_ids,
                                                                    DAUGHTER_COLORS)},
                                              None:(0,0,0,0)}
    
    ################ Mouse click events ###############
    
    @log_error
    def _track_clicked_wrapper(self, viewer, event):
        if self.state == ViewerState.SELECT_REGION:
            return # Does nothing if the state is "select region"
        logger.info(event.modifiers)
        logger.info("Track clicked")
        yield  # important to avoid a potential bug when selecting the daughter
        logger.info("button released")
        data_coordinates = self._cropped_labels_layer.world_to_data(event.position)
        logger.debug(f"world coordinates: {event.position}")
        logger.debug(f"data coordinates: {data_coordinates}")
        #track_id = find_track_by_coordinates(self._gav._spatial_filter, data_coordinates)
        track_id = int(np.asarray(self._track_gav[self.window][tuple([int(round(c)) for c in data_coordinates])]))
        if track_id == 0:
            logger.info("No track found.")
            return
        logger.info(f"Track ID: {track_id}")
        self.track_clicked(track_id) # trigger the state machine transition
#            if "Control" in event.modifiers:
#                logger.info("Control pressed, removing from the verified or candidate list")
#                self.verified_track_ids.discard(int(val))
#                self.candidate_track_ids.discard(int(val))
#                self.update_finalized_point_layer()
#                self._write_verified_and_candidates()
#            else:
#                frame = coords[0]
#                logger.info(f"clicked at {coords} at frame {frame} and label value {val}")
#                
        
    @log_error
    def _region_clicked_wrapper(self, viewer, event):
        if self.state != ViewerState.SELECT_REGION:
            return # Does nothing if the state is not "select region"
        logger.info("Region clicked")
        data_coordinates = self._labels_layer.world_to_data(event.position)
        cords = np.round(data_coordinates).astype(int)
        logger.info(f"clicked at {cords}")
        self.region_clicked(cords)
    
    ################ Select region ################
    @log_error
    def crop_region(self,coords):
        logger.info(f"Region selected: coords {coords}")
        window = (slice(None),
                  slice(max(0,coords[1]-self.crop_size//2),
                        min(coords[1]+self.crop_size//2,self._track_gav.shape[-2])), 
                  slice(max(0,coords[2]-self.crop_size//2),
                        min(coords[2]+self.crop_size//2,self._track_gav.shape[-1])))
        self.window = window
        logger.info(f"window selected: {window}")
        translate = [0]+[s.start for s in window[1:]]
        self._cropped_image_layer.data = self._image[window]
        self._cropped_image_layer.translate = translate
        cropped_label = self._track_gav[window]
        self._cropped_labels_layer.data = cropped_label
        self._cropped_labels_layer.translate = translate
        cropped_verified = self._verified_gav[window]
        self._cropped_verified_layer.data = cropped_verified
        self._cropped_verified_layer.translate = translate

    ################ Common utilities ################
    @log_error
    def _get_frame_and_tracklet_frames_min_max(self):
        assert self._selected_track is not None, "No selected track."
        iT = self._viewer.dims.current_step[0]
        track_frames = self._selected_track.frames
        return iT, min(track_frames), max(track_frames)
    
    @log_error
    def _get_mask_from_redraw_layer(self):
        inds = np.nonzero(self._redraw_layer.data == 1)
        min_y, min_x, max_y, max_x = inds[0].min(), inds[1].min(), inds[0].max(), inds[1].max()
        mask_data = np.array(self._redraw_layer.data[min_y:max_y+1, min_x:max_x+1] == 1)
        mask = td.nodes.Mask(mask=mask_data, bbox=(min_y, min_x, max_y+1, max_x+1))
        return mask
    
    @log_error
    def _reselect_from_node_id(self, node_id):
        new_track_id = self._tracks.graph.filter(
            node_ids=[node_id]
        ).node_attrs(attr_keys=[self.tracklet_id_attr_name])[self.tracklet_id_attr_name].to_list()[0]
            
        self._selected_track = _SelectedTrackInfo(
            graph=self._tracks.graph,
            track_id=new_track_id,
            tracklet_id_attr_name=self.tracklet_id_attr_name
        )
    
    ################ Select tracks, finalize and abort edits ################

    @log_error    
    def select_track(self, track_id):
        logger.info(f"Selecting track ID: {track_id}")
        self._selected_track = _SelectedTrackInfo(
            graph=self._tracks.graph,
            track_id=track_id,
            tracklet_id_attr_name=self.tracklet_id_attr_name
        )
        logger.info(f"Track ID: {track_id}, Successor IDs: {self._selected_track.daughter_track_ids}")
        self._cropped_labels_layer.selected_label = track_id

    @log_error    
    def finalize_track(self):
        assert self._selected_track is not None, "No selected track."
        logger.info("Track finalized")
        self._selected_track.subgraph.update_node_attrs(attrs={
            "verified":True
        })
        #self.verified_track_ids.add(int(self._selected_track.track_id))
        # XXX implement removal of previous daughters from candidate list
#        for track_id in self.original_splits.get(int(self._selected_track.tra), []):
#            logger.info(f"Previous daughter {track_id} removed from the candidate list")
#            self.candidate_track_ids.discard(int(track_id))
        #self.candidate_track_ids.discard(int(self._selected_track.track_id))
        #self.candidate_track_ids.update(map(int,set(self._selected_track.daughter_track_ids)
        #                                    -set(self.verified_track_ids)))

        ## XXX : Maybe in another thread? Make it sure that this happens surely independent of the main thread exit, and in the correct order.
        #self.ta.attrs["verified_track_ids"] = list(self.verified_track_ids)
        #self.ta.attrs["candidate_track_ids"] = list(self.candidate_track_ids)
        #self.ta.write_properties()   
        #self.txn.commit_sync()
        
        #self.update_finalized_point_layer()
    
    @log_error    
    def abort_transaction(self):
        logger.info("Transaction aborted")
        logger.warning("Transaction cannot be aborted for now.")
        
    ################ Redraw labels ################
    @log_error
    def label_redraw_enter_valid(self):
        assert self._selected_track is not None, "No selected track."
        iT, frames_min, frames_max = self._get_frame_and_tracklet_frames_min_max()
        is_valid =  (iT >= frames_min-1) and (iT <= frames_max+1)
        if not is_valid:
            logger.info("track does not exist in connected timeframe")
        return is_valid

    @log_error
    def prepare_redraw_layer(self):
        logger.info("Prepare redraw layer")
        assert self._selected_track is not None, "No selected track."
        self._redraw_layer.data = np.zeros_like(self._redraw_layer.data)
        iT = self._viewer.dims.current_step[0]
        cropped_label_frame = np.asarray(self._track_gav[self.window][iT])
        self._redraw_layer.data[self.window[1:]] = (cropped_label_frame == self._selected_track.track_id).astype(bool)

    @log_error
    def check_drawn_label(self):
        return np.any(self._redraw_layer.data == 1)
 
    @log_error
    def label_redraw_finish(self):
        logger.info("label redraw finish")
        assert self._selected_track is not None, "No selected track."
        iT = self._viewer.dims.current_step[0]
        node_id = self._selected_track.subgraph.filter(
            td.NodeAttr("t") == iT
        ).node_ids()
        if len(node_id) == 0:
            node_id = None
            connected_node_ids = self._selected_track.subgraph.filter(
                td.NodeAttr("t") == (iT - 1 if iT > min(self._selected_track.frames) else iT + 1)
            ).node_ids()
            if len(connected_node_ids) == 0:
                logger.error("No connected node found at the adjacent frame.")
                connected_node_id = None
            elif len(connected_node_ids) > 1:
                logger.error("Multiple connected nodes found at the adjacent frame, using the first one.")
                connected_node_id = connected_node_ids[0]
            else:
                connected_node_id = connected_node_ids[0]
        else:
            if len(node_id) > 1:
                logger.error("Multiple nodes found at the current frame, using the first one.")
            node_id = node_id[0]
            connected_node_id = None
        
        mask = self._get_mask_from_redraw_layer()
        if node_id is not None:
            #if ask_draw_label(self._viewer) == "new":
            # XXX Currently adding new mask without modifying the old one is not supported
            logger.info("Redrawing existing mask")
            self._tracks.apply(
                action.RedrawMaskAction(
                    node_id=node_id,
                    new_mask=mask
                )
            )
        else:
            logger.info("Adding new mask")
            self._tracks.apply(
                action.AddNodeAction(
                    frame=iT,
                    mask=mask,
                    connected_node_id=connected_node_id,
                    tracklet_id=self._selected_track.track_id
            ))
            self._selected_track.refresh()
        
    ################ Switch tracks ################
    @log_error
    def switch_track_enter_valid(self):
        iT, frames_min, frames_max = self._get_frame_and_tracklet_frames_min_max()
        is_valid =  (iT >= frames_min-1) and (iT <= frames_max+1)
        if not is_valid:
            logger.info("track does not exist in connected timeframe")
        return is_valid
    
    @log_error
    def switch_track(self, switch_target_label):
        assert self._selected_track is not None, "No selected track."
        frame, frames_min, frames_max = self._get_frame_and_tracklet_frames_min_max()
        direction = choose_direction_by_mbox(self._viewer)
        if not direction:
            logger.info("Switch cancelled")
            return
        elif direction == "forward":
            if frame <= frames_min:
                logger.info("Cannot switch forward, since the track does not exist in the previous frame")
                return
            logger.info("Switching forward")
            node_id1 = self._selected_track.subgraph.filter(
                td.NodeAttr("t") == frame-1
            ).node_ids()[0]
            node_id2 = self._tracks.graph.filter(
                td.NodeAttr(self.tracklet_id_attr_name) == switch_target_label,
                td.NodeAttr("t") == frame
            ).node_ids()[0]
            track_node_id = node_id1
        elif direction == "backward":
            if frame >= frames_max:
                logger.info("Cannot switch backward, since the track does not exist in the next frame")
                return
            logger.info("Switching backward")
            node_id2 = self._selected_track.subgraph.filter(
                        td.NodeAttr("t") == frame+1
            ).node_ids()[0]
            node_id1 = self._tracks.graph.filter(
                        td.NodeAttr(self.tracklet_id_attr_name) == switch_target_label,
                        td.NodeAttr("t") == frame
            ).node_ids()[0]
            track_node_id = node_id2
        self._tracks.apply(
            action.ConnectTrackAction(
                node_id1=node_id1,
                node_id2=node_id2
            )
        )
        self._reselect_from_node_id(track_node_id)
        logger.info("Tracks switched")

    ################ Mark termination ################
    @log_error
    def mark_termination_enter_valid(self):
        frame, frames_min, frames_max = self._get_frame_and_tracklet_frames_min_max()
        is_valid =  (frame >= frames_min) and (frame <= frames_max)
        if not is_valid:
            logger.info("track does not exist in the current timeframe")
        return is_valid
    
    @log_error
    def mark_termination(self):
        assert self._selected_track is not None
        iT = self._viewer.dims.current_step[0]
        node_id = self._selected_track.subgraph.filter(
            td.NodeAttr("t") == iT
        ).node_ids()[0]
        annotation, res = get_annotation_of_track_end(
            self._viewer, ""
        )
        if res:
            self._tracks.apply(
                action.AnnotateTerminationAction(
                    node_id=node_id,
                    termination_annotation=annotation,
                    delete_successor_tracklet=False #TODO ask this to user
                )
            )
            self._reselect_from_node_id(node_id)
        else:
            logger.info("Mark termination cancelled")

    ################ Daughter selection ################
    @log_error
    def daughter_choose_mode_enter_valid(self):
        frame, frames_min, frames_max = self._get_frame_and_tracklet_frames_min_max()
        is_valid =  (frame >= frames_min) and (frame <= frames_max+1)
        if not is_valid:
            logger.info("track does not exist in the current timeframe")
        return is_valid
        
    @log_error
    def reset_daughter(self):
        self._daughter_candidates = []
        self._daughter_frame = self._viewer.dims.current_step[0]
 
    @log_error
    def on_enter_DAUGHTER_CHOOSE_MODE(self, *_):
        logger.info("Current daughter candidates count: %i", len(self._daughter_candidates))
        if len(self._daughter_candidates) == 2:
            self.finalize_daughter()
            self.to_LABEL_SELECTED()
        else:
            method = choose_division_by_mbox(self._viewer)
            logger.info("%s selected", method)
            if method == "select":
                self.to_DAUGHTER_SWITCH()
            elif method == "draw":
                self.to_DAUGHTER_DRAW()
            else:
                self.to_LABEL_SELECTED()

    @log_error
    def daughter_select(self, daughter_label):
        frame = self._viewer.dims.current_step[0]
        logger.info(f"Selecting daughter at frame {frame} with label {daughter_label}")
        if frame == self._daughter_frame:
            node_id = self._tracks.graph.filter(
                td.NodeAttr(self.tracklet_id_attr_name) == daughter_label,
                td.NodeAttr("t") == frame
            ).node_ids()
            self._daughter_candidates.append(node_id[0])
            logger.info("daughter selected.")
        else:
            logger.info("frame not correct. daughter not selected.")

    @log_error
    def daughter_draw_finish(self):
        frame = self._viewer.dims.current_step[0]
        logger.info("label redraw finish")
        mask = self._get_mask_from_redraw_layer()
        logger.info("add mask")
        self._daughter_candidates.append((frame, mask))

    @log_error
    def finalize_daughter(self):
        assert self._selected_track is not None, "No selected track."
        assert len(self._daughter_candidates) == 2
        logger.info("finalize daughter")
        node_id = self._selected_track.subgraph.filter(
            td.NodeAttr("t") == self._viewer.dims.current_step[0] - 1
        ).node_ids()[0]
        logger.info(f"parent node id: {node_id}")
        self._tracks.apply(
            action.AnnotateDaughterAction(
                node_id=node_id,
                daughters=self._daughter_candidates
            )
        )
        self._reselect_from_node_id(node_id)
        logger.info("daughters finalized.")
        
