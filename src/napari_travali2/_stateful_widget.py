from transitions import Machine
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from magicgui.widgets import Container, create_widget, Label
from napari import Viewer
from napari.layers import Labels as LabelsLayer
from ._transitions import ViewerState, TRANSITIONS
from ._logging import logger, log_error
import numpy as np

class StateMachineWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer", ):
        super().__init__()

        self._viewer = viewer
        self._layer_select = create_widget(annotation=LabelsLayer, label="Select a label layer")
        # Set up the layout

        # Add a label to show the current state
        self._state_label = Label(value=f"Current State: {ViewerState.ALL_LABEL}")
        #self._state_label.setText(f"Current State: {ViewerState.ALL_LABEL}")
        self.extend([
            self._layer_select,
            self._state_label
        ])
        
        self.states = ["idle", "running", "paused"]
        self.machine = Machine(model=self, 
                               states=ViewerState, 
                               initial=ViewerState.ALL_LABEL,
                               transitions=TRANSITIONS,
                               after_state_change="update_viewer_status",
                               ignore_invalid_triggers=True)
        

        self._viewer.bind_key("Enter", lambda _viewer: self.Enter_typed(), overwrite=True)
        self._viewer.bind_key("r", lambda _viewer: self.r_typed(), overwrite=True)
        self._viewer.bind_key("s", lambda _viewer: self.s_typed(), overwrite=True)
        self._viewer.bind_key("d", lambda _viewer: self.d_typed(), overwrite=True)
        self._viewer.bind_key("t", lambda _viewer: self.t_typed(), overwrite=True)
        self._viewer.bind_key("Escape", lambda _viewer: self.Escape_typed(), overwrite=True)
        # getting the current layer from the layer_select
        self._layer_select.changed.connect(self.layer_changed)
        self.layer_changed(self._layer_select.value)
        
        #layer = 
        
    def layer_changed(self, layer):
        self.label_layer = layer
        logger.info("Layer changed")
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
        self.label_layer.mouse_drag_callbacks.append(track_clicked)
        
    def update_viewer_status(self):
        print(f"Current State: {self.state}")
        #self._state_label.value = f"Current State: {self.state}"

