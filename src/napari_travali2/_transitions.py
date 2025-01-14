from enum import Enum

class ViewerState(Enum):
    ALL_LABEL = 1
    LABEL_SELECTED = 2
    LABEL_REDRAW = 3
    LABEL_SWITCH = 4
    DAUGHTER_SWITCH = 5
    DAUGHTER_DRAW = 6
    DAUGHTER_CHOOSE_MODE = 7

TRANSITIONS = [
    {
        #"trigger": "track_clicked",
        "trigger": "Enter_typed",
        "source": ViewerState.ALL_LABEL,
        "dest": ViewerState.LABEL_SELECTED,
        #"before": "select_track",
    },
    {
        "trigger": "Escape_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.ALL_LABEL,
    },
    {
        "trigger": "Enter_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.ALL_LABEL,
        #"before": "finalize_track",
    },
    {
        "trigger": "r_typed",
        "source": ViewerState.LABEL_SELECTED,
        "dest": ViewerState.LABEL_REDRAW,
        #"conditions": "label_redraw_enter_valid",
        #"before": "refresh_redraw_label_layer",
    },
    {
        "trigger": "Enter_typed",
        "source": ViewerState.LABEL_REDRAW,
        "dest": ViewerState.LABEL_SELECTED,
        #"conditions": "check_drawn_label",
        #"before": "label_redraw_finish",
    },
    {
        "trigger": "Escape_typed",
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
        "trigger": "Escape_typed",
        "source": ViewerState.LABEL_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": "track_clicked",
        "source": ViewerState.LABEL_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
        #"before": "switch_track",
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
        "trigger": "Enter_typed",
        "source": ViewerState.DAUGHTER_DRAW,
        "dest": ViewerState.DAUGHTER_CHOOSE_MODE,
        #"conditions": "check_drawn_label",
        #"before": "daughter_draw_finish",
    },
    {
        "trigger": "Escape_typed",
        "source": ViewerState.DAUGHTER_SWITCH,
        "dest": ViewerState.LABEL_SELECTED,
    },
    {
        "trigger": "Escape_typed",
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
