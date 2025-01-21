from transitions import Machine
from magicgui.widgets import Container, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS, STATE_EXPLANATION
from ._logging import logger, log_error
from ._gui_utils import choose_direction_by_mbox, get_annotation_of_track_end,choose_division_by_mbox, ask_draw_label
from ._consts import SELECTED_COLOR, DAUGHTER_COLORS
import numpy as np
import tensorstore as ts
from copy import deepcopy
import trackarray_tensorstore as tats
from napari.utils.colormaps import label_colormap

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

VIEWER_STATE_VISIBILITY = { 
                           # [label_layer, redraw_layer]
    ViewerState.ALL_LABEL:      [True, False],
    ViewerState.LABEL_SELECTED: [True, False],
    ViewerState.LABEL_REDRAW:   [False, True],
    ViewerState.LABEL_SWITCH:   [True, False],
    ViewerState.DAUGHTER_SWITCH:[True, False],
    ViewerState.DAUGHTER_DRAW:  [False, True],
    ViewerState.DAUGHTER_CHOOSE_MODE: [True, False],
}

class StateMachineWidget(Container):
    def __init__(self, 
                 viewer: Viewer, 
                 ta: tats.TrackArray, 
                 image_data,
                 crop_size=1024):
        super().__init__()

        self._viewer = viewer
        self._image_layer = viewer.add_image([image_data, image_data[::2,::2]], name="Image")
        self._label_layer = viewer.add_labels([ta.array, ta.array[::2,::2]], name="Labels")
        self.crop_size = crop_size

        shape = ta.array.shape[:-2]
        cropped_shape = (*shape, crop_size, crop_size)
        if "Cropped Image" in viewer.layers:
            viewer.layers.remove(viewer.layers["Cropped Image"])
        self._cropped_image_layer = viewer.add_image(
            np.zeros(cropped_shape, dtype=image_data.dtype.name), 
            name="Cropped Image")
        
        if "Cropped Labels" in viewer.layers:
            viewer.layers.remove(viewer.layers["Cropped Labels"])
        self._cropped_label_layer = viewer.add_labels(
            np.zeros(cropped_shape, dtype=ta.array.dtype.name), 
            name="Cropped Labels")
        
        if "Redraw" in viewer.layers:
            viewer.layers.remove(viewer.layers["Redraw"])
        self._redraw_layer = viewer.add_labels(
            np.zeros(ta.array.shape[-2:],dtype=bool), 
            name="Redraw", cache=False)
       
        self.txn = None
        self.ta = ta
        self.image_data = image_data

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
        

        # XXX possibly refactor this
        self._viewer.bind_key("o", lambda event: self.o_typed(), overwrite=True)
        self._viewer.bind_key("r", lambda event: self.r_typed(), overwrite=True)
        self._viewer.bind_key("s", lambda event: self.s_typed(), overwrite=True)
        self._viewer.bind_key("d", lambda event: self.d_typed(), overwrite=True)
        self._viewer.bind_key("t", lambda event: self.t_typed(), overwrite=True)
        self._viewer.bind_key("n", lambda event: self.n_typed(), overwrite=True)
        self._viewer.bind_key("c", lambda event: self.c_typed(), overwrite=True)

        self.update_viewer_status()
        
        @log_error
        def track_clicked(layer, event):
            logger.info("Track clicked")
            yield  # important to avoid a potential bug when selecting the daughter
            logger.info("button released")
            data_coordinates = layer.world_to_data(event.position)
            logger.debug(f"world coordinates: {event.position}")
            logger.debug(f"data coordinates: {data_coordinates}")
            coords = np.round(data_coordinates).astype(int)
            val = layer.get_value(data_coordinates)
            try:
                _ = iter(val)
                val = val[-1]
            except TypeError as te:
                pass
                
            if val is None:
                return
            if val != 0:
                frame = coords[0]
                logger.info(f"clicked at {coords} at frame {frame} and label value {val}")
                self.track_clicked(frame, val)
        self._cropped_label_layer.mouse_drag_callbacks.append(track_clicked)
        
        @log_error
        def region_clicked(layer, event):
            logger.info("Region clicked")
            data_coordinates = layer.world_to_data(event.position)
            cords = np.round(data_coordinates).astype(int)
            logger.info(f"clicked at {cords}")
            self.region_clicked(cords)
        self._label_layer.mouse_drag_callbacks.append(region_clicked)
    
    @log_error    
    def update_viewer_status(self,*_args):
        self._state_label.value = ("========================\n"
                                  f"Current State: {self.state.name}\n"
                                  "========================\n"
                                  "\n"
                                  f"{STATE_EXPLANATION[self.state]}")
        if self.state == ViewerState.SELECT_REGION:
            self._label_layer.visible = True
            self._image_layer.visible = True
            self._cropped_image_layer.visible = False
            self._cropped_label_layer.visible = False
            self._redraw_layer.visible = False
            self._viewer.layers.selection.active = self._label_layer
        else:
            self._label_layer.visible = False
            self._image_layer.visible = False
            self._cropped_image_layer.visible = True
            visibility = VIEWER_STATE_VISIBILITY[self.state]
            for layer, visible in zip([self._cropped_label_layer, self._redraw_layer], visibility):
                layer.visible = visible
            active_layer = self._cropped_label_layer if visibility[0] else self._redraw_layer
            self._viewer.layers.selection.active = active_layer

            if self.state not in SHOW_SELECTED_LABEL_STATES:
                self._cropped_label_layer.colormap = default_colormap()
            else:
                self.set_selected_colormap()

            if visibility[1]:
                self._redraw_layer.data = np.zeros_like(self._redraw_layer.data)
                self._redraw_layer.selected_label = 1
                self._redraw_layer.mode = "paint"
                # XXX better if I can set the viewer.dims not to change
    
    @log_error
    def set_selected_colormap(self):
        self._cropped_label_layer.colormap = {0:(0,0,0,0), 
                                              self._selected_label:SELECTED_COLOR,
                                              **{d:c for d,c in zip(self._daughters,DAUGHTER_COLORS)},
                                              None:(0,0,0,0)}
    
    @log_error
    def update_daughters(self):
        self._daughters = self.ta.splits.get(self._selected_label, [])
    
    ################ Select region ################
    @log_error
    def crop_region(self,coords):
        logger.info(f"Region selected: coords {coords}")
        window = (slice(None),
                  slice(coords[1]-self.crop_size//2,coords[1]+self.crop_size//2), 
                  slice(coords[2]-self.crop_size//2,coords[2]+self.crop_size//2))
        self.window = window
        logger.info(f"window selected: {window}")
        translate = [0]+[s.start for s in window[1:]]
        self._cropped_image_layer.data = self.image_data[window][ts.d[:].translate_to[0]]
        self._cropped_image_layer.translate = translate
        cropped_label = self.ta.array[window][ts.d[:].translate_to[0]]
        self._cropped_label = cropped_label
        self._cropped_label_layer.data = cropped_label
        self._cropped_label_layer.translate = translate

    
    ################ Select tracks, finalize and abort edits ################
    @log_error    
    def select_track(self,frame,val):
        self._selected_label = val
        self.original_bboxes_dict = deepcopy(self.ta._bboxes_dict)
        self.original_splits = deepcopy(self.ta.splits)
        self.original_termination_annotations = deepcopy(self.ta.termination_annotations)
        logger.info(f"Track selected: frame {frame} value {val}")
        assert self.txn is None
        self.txn = ts.Transaction()
        array_txn = self._cropped_label.with_transaction(self.txn)
        self._cropped_label_layer.data = array_txn
        self._cropped_label_layer.selected_label = val
        self.update_daughters()

    @log_error    
    def finalize_track(self):
        logger.info("Track finalized")
        self.txn.commit_sync()
        self._cropped_label_layer.data = self._cropped_label
    
        self.txn = None
        self._cropped_label_layer.selected_label = 0
    
    @log_error    
    def abort_transaction(self):
        logger.info("Transaction aborted")
        self._cropped_label_layer.data = self._cropped_label
        self.ta._bboxes_dict = self.original_bboxes_dict
        self.ta.splits = self.original_splits
        self.ta.termination_annotations = self.original_termination_annotations
        
        self.txn.abort()
        self.txn = None
        self._cropped_label_layer.selected_label = 0
        
    ################ Redraw labels ################
    @log_error
    def label_redraw_enter_valid(self):
        iT = self._viewer.dims.current_step[0]
        track_bboxes_df = self.ta._get_track_bboxes(self._selected_label).reset_index()
        frames = track_bboxes_df["frame"].values
        is_valid =  (iT >= frames.min()-1) and (iT <= frames.max()+1)
        if not is_valid:
            logger.info("track does not exist in connected timeframe")
        return is_valid

    @log_error
    def check_drawn_label(self):
        return np.any(self._redraw_layer.data == 1)
 
    @log_error
    def label_redraw_finish(self):
        logger.info("label redraw finish")
        iT = self._viewer.dims.current_step[0]
        previous_frames = self.ta._get_track_bboxes(self._selected_label).index.get_level_values("frame").values
        
        inds = np.nonzero(self._redraw_layer.data == 1)
        min_y, min_x, max_y, max_x = inds[0].min(), inds[1].min(), inds[0].max(), inds[1].max()
        mask = self._redraw_layer.data[min_y:max_y+1, min_x:max_x+1] == 1
        
        if iT in previous_frames:
            if ask_draw_label(self._viewer) == "new":
                safe_label = self.ta._get_safe_track_id()
                logger.info("add mask")
                self.ta._update_trackid(iT, self._selected_label, safe_label, self.txn)
                self.ta.add_mask(iT, self._selected_label, (min_y, min_x), mask, self.txn)
            else:
                self.ta.update_mask(iT, self._selected_label, (min_y, min_x), mask, self.txn)
        else:
            self.ta.add_mask(iT, self._selected_label, (min_y, min_x), mask, self.txn)
        
    ################ Switch tracks ################
    @log_error
    def switch_track_enter_valid(self):
        iT = self._viewer.dims.current_step[0]
        track_bboxes_df = self.ta._get_track_bboxes(self._selected_label).reset_index()
        frames = track_bboxes_df["frame"].values
        is_valid =  (iT >= frames.min()-1) and (iT <= frames.max()+1)
        if not is_valid:
            logger.info("track does not exist in connected timeframe")
        return is_valid
    
    @log_error
    def switch_track(self, frame, switch_target_label):
        track_bboxes_df = self.ta._get_track_bboxes(self._selected_label).reset_index()
        frames = track_bboxes_df["frame"].values
        direction = choose_direction_by_mbox(self._viewer)
        if not direction:
            logger.info("Switch cancelled")
            return
        elif direction == "forward":
            if frame <= frames.min():
                logger.info("Cannot switch forward, since the track does not exist in the previous frame")
                return
            logger.info("Switching forward")
            self.ta.break_track(frame, self._selected_label, True, self.txn)
            self.ta.break_track(frame, switch_target_label, True, self.txn, 
                                new_trackid=self._selected_label)
        elif direction == "backward":
            if frame >= frames.max():
                logger.info("Cannot switch backward, since the track does not exist in the next frame")
                return
            logger.info("Switching backward")
            self.ta.break_track(frame+1, self._selected_label, False, self.txn)
            self.ta.break_track(frame+1, switch_target_label, False, self.txn, 
                                new_trackid=self._selected_label)
        self.ta.cleanup_single_daughter_splits()
        

    ################ Mark termination ################
    @log_error
    def mark_termination_enter_valid(self):
        iT = self._viewer.dims.current_step[0]
        track_bboxes_df = self.ta._get_track_bboxes(self._selected_label).reset_index()
        frames = track_bboxes_df["frame"].values
        is_valid =  (iT >= frames.min()) and (iT <= frames.max())
        if not is_valid:
            logger.info("track does not exist in the current timeframe")
        return is_valid
    
    @log_error
    def mark_termination(self):
        iT = self._viewer.dims.current_step[0]
        annotation, res = get_annotation_of_track_end(
            self._viewer, self.ta.termination_annotations.get(self._selected_label, "")
        )
        if res:
            self.ta.terminate_track(iT, self._selected_label, annotation, self.txn)
        else:
            logger.info("Mark termination cancelled")

    ################ Daughter selection ################
    @log_error
    def daughter_choose_mode_enter_valid(self):
        iT = self._viewer.dims.current_step[0]
        track_bboxes_df = self.ta._get_track_bboxes(self._selected_label).reset_index()
        frames = track_bboxes_df["frame"].values
        is_valid =  (iT >= frames.min()) and (iT <= frames.max()+1)
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
    def daughter_select(self, frame, daughter_label):
        if frame == self._daughter_frame:
            self._daughter_candidates.append(daughter_label)
        else:
            logger.info("frame not correct")

    @log_error
    def daughter_draw_finish(self):
        iT = self._viewer.dims.current_step[0]
        logger.info("label redraw finish")
        inds = np.nonzero(self._redraw_layer.data == 1)
        min_y, min_x, max_y, max_x = inds[0].min(), inds[1].min(), inds[0].max(), inds[1].max()
        mask = self._redraw_layer.data[min_y:max_y+1, min_x:max_x+1] == 1
        safe_label = self.ta._get_safe_track_id()
        logger.info("add mask")
        self.ta.add_mask(iT, safe_label, (min_y, min_x), mask, self.txn)
        self._daughter_candidates.append(safe_label)

    @log_error
    def finalize_daughter(self):
        assert len(self._daughter_candidates) == 2
        self.ta.add_split(self._daughter_frame,
                          self._selected_label, 
                          self._daughter_candidates, 
                          self.txn)
        self.update_daughters()
