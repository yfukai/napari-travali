from transitions import Machine
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from magicgui.widgets import Container, create_widget, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS, STATE_EXPLANATION
from ._logging import logger, log_error
import numpy as np
import tensorstore as ts
import tensorstore_trackarr as tta

SHOW_SELECTED_LABEL_STATES = [
    ViewerState.LABEL_SELECTED,
    ViewerState.LABEL_REDRAW,
    ViewerState.DAUGHTER_CHOOSE_MODE,
    ViewerState.DAUGHTER_DRAW,
    ViewerState.DAUGHTER_SWITCH,
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
        
            
        if self.state == ViewerState.LABEL_REDRAW:
            self._redraw_layer.data = np.zeros_like(self._redraw_layer.data)
            self._redraw_layer.selected_label = 1
            self._redraw_layer.mode = "paint"
            # TODO better if I can set the viewer.dims not to change
        
    @log_error    
    def select_track(self,frame,val):
        self._selected_label = val
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
        self.txn.abort()
        self.txn = None
        self._label_layer.selected_label = 0
        
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
        
    