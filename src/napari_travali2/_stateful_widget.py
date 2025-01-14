from transitions import Machine
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.types import LabelsData
from ._transitions import ViewerState, TRANSITIONS
from ._logging import logger, log_error
import numpy as np

class StateMachineWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()

        self.viewer = viewer
        self.layer_select = create_widget(annotation=LabelsData, label="Select a label layer")
        # Set up the layout
        self.layout = QVBoxLayout()

        # Add a label to show the current state
        self.state_label = QLabel("Current State: before initialization.")
        
        self.layout.addWidget(self.layer_select.native)
        self.layout.addWidget(self.state_label)
        self.setLayout(self.layout)

        # Define the state machine
        self.states = ["idle", "running", "paused"]
        self.machine = Machine(model=self, 
                               states=ViewerState, 
                               initial=ViewerState.ALL_LABEL,
                               transitions=TRANSITIONS,
                               after_state_change="update_viewer_status",
                               ignore_invalid_triggers=True)
        

        self.viewer.bind_key("Enter", lambda _viewer: self.Enter_typed(), overwrite=True)
        self.viewer.bind_key("r", lambda _viewer: self.r_typed(), overwrite=True)
        self.viewer.bind_key("s", lambda _viewer: self.s_typed(), overwrite=True)
        self.viewer.bind_key("d", lambda _viewer: self.d_typed(), overwrite=True)
        self.viewer.bind_key("t", lambda _viewer: self.t_typed(), overwrite=True)
        self.viewer.bind_key("Escape", lambda _viewer: self.Escape_typed(), overwrite=True)
        # getting the current layer from the layer_select
        self.layer_select.changed.connect(self.layer_changed)
        print(self.layer_select)
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
                msg = f"clicked at {cords}"
                frame = cords[0]
                logger.info("%s %i %s", msg, frame, val)
                self.track_clicked(frame, val)
        self.label_layer.mouse_drag_callbacks.append(track_clicked)
        
    def update_viewer_status(self):
        self.state_label.setText(f"Current State: {self.state}")

