from enum import Enum

class ViewerState(Enum):
    ALL_LABEL = 1
    LABEL_SELECTED = 2
    LABEL_REDRAW = 3
    LABEL_SWITCH = 4
    DAUGHTER_SWITCH = 5
    DAUGHTER_DRAW = 6
    DAUGHTER_CHOOSE_MODE = 7

Enter_key = "o"
Escape_key = "n"

TRANSITIONS = [
    {
        "trigger": "track_clicked",
        "source": ViewerState.ALL_LABEL,
        "dest": ViewerState.LABEL_SELECTED,
        "before": "select_track",
    },
    {
        "trigger": f"{Escape_key}_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.ALL_LABEL,
        "before": "abort_transaction",
    },
    {
        "trigger": f"{Enter_key}_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.ALL_LABEL,
        "before": "finalize_track",
    },
    {
        "trigger": "r_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.LABEL_REDRAW,
        "conditions": "label_redraw_enter_valid",
    },
    {
        "trigger": f"{Enter_key}_typed",
        "source": ViewerState.LABEL_REDRAW,
        "dest": ViewerState.LABEL_SELECTED,
        "conditions": "check_drawn_label",
        "before": "label_redraw_finish",
    },
    {
        "trigger": f"{Escape_key}_typed",
        "source": ViewerState.LABEL_REDRAW,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": "s_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.LABEL_SWITCH,
        #"conditions": "switch_track_enter_valid",
    },
    {
        "trigger": f"{Escape_key}_typed",
        "source": ViewerState.LABEL_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": "track_clicked",
        "source": ViewerState.LABEL_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
        "before": "switch_track",
    },
    {
        "trigger": "d_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.DAUGHTER_CHOOSE_MODE,
        #"conditions": "daughter_choose_mode_enter_valid",
    },
    {
        "trigger": "track_clicked",
        "source": ViewerState.DAUGHTER_SWITCH,
        "dest": ViewerState.DAUGHTER_CHOOSE_MODE,
        #"before": "daughter_select",
    },
    {
        "trigger": f"{Enter_key}_typed",
        "source": ViewerState.DAUGHTER_DRAW,
        "dest": ViewerState.DAUGHTER_CHOOSE_MODE,
        #"conditions": "check_drawn_label",
        #"before": "daughter_draw_finish",
    },
    {
        "trigger": f"{Escape_key}_typed",
        "source": ViewerState.DAUGHTER_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": f"{Escape_key}_typed",
        "source": ViewerState.DAUGHTER_DRAW,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": "t_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.LABEL_SELECTED,
        #"conditions": "mark_termination_enter_valid",
        #"before": "mark_termination",
    },
]

STATE_EXPLANATION = {
    ViewerState.ALL_LABEL: "All labels are shown.\nPlease click on a track to select it.",
    ViewerState.LABEL_SELECTED: "A track is selected.\n"
                      f"Press '{Enter_key}' to finalize the selection.\n"
                      "Press 'r' to redraw the label mask of the frame.\n"
                      "Press 's' to switch the tracks.\n"
                      "Press 'd' to select the daughters.\n"
                      "Press 't' to mark the termination of the track.\n"
                     f"Press '{Escape_key}' to deselect the track.",
    ViewerState.LABEL_REDRAW: "Redrawing the label mask.\n"
                             f"Press '{Enter_key}' to finish.",
    ViewerState.LABEL_SWITCH: "Switching the tracks.\n"
                              "Click on the track to switch to.\n"
                             f"Press '{Escape_key}' to cancel.",
    ViewerState.DAUGHTER_CHOOSE_MODE: "Select or draw the daughter.\n",
    ViewerState.DAUGHTER_SWITCH: "Switching the daughter.\n"
                                "Click on the daughter to switch to.\n"
                               f"Press '{Escape_key}' to cancel.",
    ViewerState.DAUGHTER_DRAW: "Drawing the daughter.\n"
                              f"Press '{Enter_key}' to finish.\n"
                              f"Press '{Escape_key}' to cancel.",
}
    