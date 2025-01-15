from transitions import Machine
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from magicgui.widgets import Container, create_widget, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS, STATE_EXPLANATION
from ._logging import logger, log_error
from ._gui_utils import choose_direction_by_mbox, get_annotation_of_track_end,choose_division_by_mbox
import numpy as np
import tensorstore as ts
from copy import deepcopy
import tensorstore_trackarr as tta

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
    def __init__(self, viewer: Viewer, label_layer: LabelsLayer, ta: tta.TrackArray):
        super().__init__()

        self._viewer = viewer
        self._label_layer = label_layer
        if "Redraw" in viewer.layers:
            viewer.layers.remove(viewer.layers["Redraw"])
        self._redraw_layer = viewer.add_labels(
            np.zeros(ta.array.shape[-2:],dtype=bool), 
            name="Redraw", cache=False)
        self.txn = None
        self.ta = ta

        # Add a label to show the current state
        self._state_label = Label(value=f"Current State: {ViewerState.ALL_LABEL}")
        #self._state_label.setText(f"Current State: {ViewerState.ALL_LABEL}")
        self.extend([
            self._state_label
        ])
        
        self.machine = Machine(model=self, 
                               states=ViewerState, 
                               initial=ViewerState.ALL_LABEL,
                               transitions=TRANSITIONS,
                               after_state_change="update_viewer_status",
                               ignore_invalid_triggers=True)
        
        self._viewer.bind_key("o", lambda event: self.o_typed(), overwrite=True)
        self._viewer.bind_key("r", lambda event: self.r_typed(), overwrite=True)
        self._viewer.bind_key("s", lambda event: self.s_typed(), overwrite=True)
        self._viewer.bind_key("d", lambda event: self.d_typed(), overwrite=True)
        self._viewer.bind_key("t", lambda event: self.t_typed(), overwrite=True)
        self._viewer.bind_key("n", lambda event: self.n_typed(), overwrite=True)
        # getting the current layer from the layer_select

        self.update_viewer_status()
        
        @log_error
        def track_clicked(layer, event):
            logger.info("Track clicked")
            yield  # important to avoid a potential bug when selecting the daughter
            logger.info("button released")
            data_coordinates = layer.world_to_data(event.position)
            cords = np.round(data_coordinates).astype(int)
            val = layer.get_value(data_coordinates)
            if val is None:
                return
            if val != 0:
                frame = cords[0]
                logger.info(f"clicked at {cords} at frame {frame} and label value {val}")
                self.track_clicked(frame, val)
        self._label_layer.mouse_drag_callbacks.append(track_clicked)
    
    @log_error    
    def update_viewer_status(self,*_args):
        self._state_label.value = ("========================\n"
                                  f"Current State: {self.state.name}\n"
                                  "========================\n"
                                  "\n"
                                  f"{STATE_EXPLANATION[self.state]}")
        if self.state in SHOW_SELECTED_LABEL_STATES:
            self._label_layer.show_selected_label = True
        else:
            self._label_layer.show_selected_label = False
            
        visibility = VIEWER_STATE_VISIBILITY[self.state]
        for layer, visible in zip([self._label_layer, self._redraw_layer], visibility):
            layer.visible = visible
        active_layer = self._label_layer if visibility[0] else self._redraw_layer
        self._viewer.layers.selection.active = active_layer
            
        if visibility[1]:
            self._redraw_layer.data = np.zeros_like(self._redraw_layer.data)
            self._redraw_layer.selected_label = 1
            self._redraw_layer.mode = "paint"
            # TODO better if I can set the viewer.dims not to change
        
    ################ Select tracks, finalize and abort edits ################
    @log_error    
    def select_track(self,frame,val):
        self._selected_label = val
        self.original_bboxes_df = self.ta.bboxes_df.copy()
        self.original_splits = deepcopy(self.ta.splits)
        self.original_termination_annotations = deepcopy(self.ta.termination_annotations)
        logger.info(f"Track selected: frame {frame} value {val}")
        assert self.txn is None
        self.txn = ts.Transaction()
        self._label_layer.data = self.ta.array.with_transaction(self.txn)
        self._label_layer.selected_label = val
        
    @log_error    
    def finalize_track(self):
        logger.info("Track finalized")
        self.txn.commit_sync()
        self._label_layer.data = self.ta.array
        self.txn = None
        self._label_layer.selected_label = 0
    
    @log_error    
    def abort_transaction(self):
        logger.info("Transaction aborted")
        self._label_layer.data = self.ta.array
        self.ta.bboxes_df = self.original_bboxes_df
        self.ta.splits = self.original_splits
        self.ta.termination_annotations = self.original_termination_annotations
        
        self.txn.abort()
        self.txn = None
        self._label_layer.selected_label = 0
        
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
        logger.info("label redraw finish")
        inds = np.nonzero(self._redraw_layer.data == 1)
        min_y, min_x, max_y, max_x = inds[0].min(), inds[1].min(), inds[0].max(), inds[1].max()
        mask = self._redraw_layer.data[min_y:max_y+1, min_x:max_x+1] == 1
        self.ta.update_mask(iT, self._selected_label, (min_y, min_x), mask, self.txn)
        
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
