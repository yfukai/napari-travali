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
        
    def select_track(self,frame,val):
        logger.info(f"Track selected: frame {frame} value {val}")
        assert self.txn is None
        self.txn = ts.Transaction()
        self._label_layer.selected_label = val
        
    def finalize_track(self):
        logger.info("Track finalized")
        self.txn.commit_sync()
        self.txn = None
        self._label_layer.selected_label = 0
    
    def abort_transaction(self):
        logger.info("Transaction aborted")
        self.txn.abort()
        self.txn = None
        self._label_layer.selected_label = 0
        
