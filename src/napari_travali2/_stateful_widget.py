from transitions import Machine
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from magicgui.widgets import Container, create_widget, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS, STATE_EXPLANATION
from ._logging import logger, log_error
import numpy as np
import tensorstore as ts

SHOW_SELECTED_LABEL_STATES = [
    ViewerState.LABEL_SELECTED,
    ViewerState.LABEL_REDRAW,
    ViewerState.DAUGHTER_CHOOSE_MODE,
    ViewerState.DAUGHTER_DRAW,
    ViewerState.DAUGHTER_SWITCH,
]

class StateMachineWidget(Container):
    def __init__(self, viewer: Viewer, label_layer: LabelsLayer):
        super().__init__()

        self._viewer = viewer
        self._label_layer = label_layer
        self.txn = None

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
        
    @log_error    
    def select_track(self,frame,val):
        logger.info(f"Track selected: frame {frame} value {val}")
        assert self.txn is None
        self.txn = ts.Transaction()
        self._label_layer.selected_label = val
        
    @log_error    
    def finalize_track(self):
        logger.info("Track finalized")
        self.txn.commit_sync()
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
        iT = self.viewer.dims.current_step[0]
        # return True if:
        # - this timeframe is in target_T
        # - segment_labels is not NOSEL_VALUE in either of this, previous, next target_T
        if not iT in self.target_Ts:
            logger.info("this frame is not in target_Ts")
            return False
        previous_iT = self.target_Ts[max(0, self.target_Ts.index(iT) - 1)]
        next_iT = self.target_Ts[
            min(len(self.target_Ts) - 1, self.target_Ts.index(iT) + 1)
        ]
        if (
            self.segment_labels[iT] == NOSEL_VALUE
            and self.segment_labels[previous_iT] == NOSEL_VALUE
            and self.segment_labels[next_iT] == NOSEL_VALUE
            #            not np.any(self.sel_label_layer.data[iT] == 1)
            #            and not np.any(self.sel_label_layer.data[min(iT + 1, self.sizeT)] == 1)
            #            and not np.any(self.sel_label_layer.data[max(iT - 1, 0)] == 1)
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("redraw valid")
            return True

    @log_error
    def check_drawn_label(self):
        return np.any(self.redraw_label_layer.data == 1)

    @log_error
    def label_redraw_finish(self):
        logger.info("label redraw finish")
        iT = self.viewer.dims.current_step[0]
        logger.info("label redraw finish")
        self.sel_label_layer.data[iT] = 0
        self.sel_label_layer.data[iT] = self.redraw_label_layer.data == 1
        self.label_edited[iT] = True
        if self.segment_labels[iT] == NOSEL_VALUE:
            self.segment_labels[iT] = NEW_LABEL_VALUE
        else:
            if ask_draw_label(self.viewer) == "new":
                self.segment_labels[iT] = NEW_LABEL_VALUE