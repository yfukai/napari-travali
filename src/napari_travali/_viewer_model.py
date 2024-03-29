from enum import Enum
from typing import Dict
from typing import List

import dask.array as da
import networkx as nx
import numpy as np
import pandas as pd
import zarr
from _settings._consts import DF_DIVISIONS_COLUMNS
from _settings._consts import DF_TRACKS_COLUMNS
from _settings._consts import NEW_LABEL_VALUE
from _settings._consts import NOSEL_VALUE
from _utils._gui_utils import ask_draw_label
from _utils._gui_utils import ask_ok_or_not
from _utils._gui_utils import choose_direction_by_mbox
from _utils._gui_utils import choose_division_by_mbox
from _utils._gui_utils import get_annotation_of_track_end
from _utils._logging import log_error
from _utils._logging import logger
from skimage.util import map_array


class ViewerState(Enum):
    ALL_LABEL = 1
    LABEL_SELECTED = 2
    LABEL_REDRAW = 3
    LABEL_SWITCH = 4
    DAUGHTER_SWITCH = 5
    DAUGHTER_DRAW = 6
    DAUGHTER_CHOOSE_MODE = 7


VIEWER_STATE_VISIBILITY = {
    ViewerState.ALL_LABEL: [True, False, False, True],
    ViewerState.LABEL_SELECTED: [True, True, False, False],
    ViewerState.LABEL_REDRAW: [False, False, True, False],
    ViewerState.LABEL_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_SWITCH: [True, False, False, True],
    ViewerState.DAUGHTER_DRAW: [False, False, True, False],
    ViewerState.DAUGHTER_CHOOSE_MODE: [False, True, False, False],
}


class ViewerModel:
    """The model responsible for updating the viewer state"""

    def __init__(
        self,
        viewer,
        target_Ts,
        label_layer,
        redraw_label_layer,
        sel_label_layer,
        finalized_label_layer,
        df_tracks: pd.DataFrame,
        df_divisions: pd.DataFrame,
        *,
        new_track_id: int,
        new_label_value: int,
        finalized_track_ids: List[int],
        candidate_track_ids: List[int],
        termination_annotations: Dict[int, str],
    ):
        """Initialize the model

        Parameters
        ----------
        travali_viewer : TravaliViewer
            the base TravaliViewer object
        df_tracks : pd.DataFrame
            a dataframe containing the segment information
        df_divisions : pd.DataFrame
            a dataframe containing the division information
        new_track_id : int
            the id of the new segment
        new_label_value : int
            the value of the new label
        finalized_track_ids : List[int]
            the list of segment ids that have been finalized
        candidate_track_ids : List[int]
            the list of segment ids that are candidates for annotation
        termination_annotations : Dict[int,str]
            the dict for association between track_id and termination annotation
        """
        self.selected_label = None
        self.track_id = None
        self.frame_childs = None
        self.label_childs = None
        self.segment_labels = None
        self.label_edited = None

        self.target_Ts = list(target_Ts)
        self.viewer = viewer
        self.label_layer = label_layer
        self.redraw_label_layer = redraw_label_layer
        self.sel_label_layer = sel_label_layer
        self.finalized_label_layer = finalized_label_layer

        self.termination_annotation = ""
        self.shape = self.label_layer.data.shape
        self.sizeT = self.label_layer.data.shape[0]

        self.df_tracks = df_tracks
        self.df_divisions = df_divisions
        self.new_track_id = new_track_id
        self.new_label_value = new_label_value
        self.finalized_track_ids = finalized_track_ids
        self.candidate_track_ids = candidate_track_ids
        self.termination_annotations = termination_annotations
        self.finalized_label_layer.data = self.label_layer.data.map_blocks(
            self.__label_to_finalized_label, dtype=np.uint8
        )

        self.viewer_state_active = {
            ViewerState.ALL_LABEL: self.label_layer,
            ViewerState.LABEL_SELECTED: self.sel_label_layer,
            ViewerState.LABEL_REDRAW: self.redraw_label_layer,
            ViewerState.LABEL_SWITCH: self.label_layer,
            ViewerState.DAUGHTER_SWITCH: self.label_layer,
            ViewerState.DAUGHTER_DRAW: self.redraw_label_layer,
            ViewerState.DAUGHTER_CHOOSE_MODE: self.sel_label_layer,
        }
        self.layers = [
            self.label_layer,
            self.sel_label_layer,
            self.redraw_label_layer,
            self.finalized_label_layer,
        ]

    @log_error
    def update_layer_status(self, *_):
        """Update the layer status according to the current viewer state."""
        visibles = VIEWER_STATE_VISIBILITY[self.state]
        assert len(visibles) == len(self.layers)
        for i in range(len(self.layers)):
            try:
                self.layers[i].visible = visibles[i]
            except ValueError:
                pass

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(self.viewer_state_active[self.state])

    @log_error
    def refresh_redraw_label_layer(self):
        """Refresh the redraw_label_layer to blank."""
        self.redraw_label_layer.data = np.zeros_like(self.redraw_label_layer.data)
        self.redraw_label_layer.mode = "paint"

    # functions to map label #
    @log_error
    def __label_to_selected_label_image(self, block, block_info=None):
        """Convert a block of label image to the selected label image."""
        #        print("block_info",block_info[0]['array-location'])
        assert not self.segment_labels is None
        assert not self.frame_childs is None
        assert not self.label_childs is None
        if block_info is None or len(block_info) == 0:
            return None
        block_location = block_info[0]["array-location"]
        iT = block_location[0][0]
        sel_label = (block == self.segment_labels[iT]).astype(np.uint8)
        # reading from df_tracks2
        for j, (frame, label) in enumerate(zip(self.frame_childs, self.label_childs)):
            if iT == frame:
                if np.isscalar(label):
                    sel_label[block == label] = j + 2
                else:
                    indices = [slice(loc[0], loc[1]) for loc in block_location]
                    sub_label = label[tuple(indices)[2:]]
                    sel_label[0, 0][sub_label] = j + 2
        return sel_label

    @log_error
    def __label_to_finalized_label(self, block, block_info=None):
        """Convert a block of label image to the finalize label image."""
        #        print("block_info",block_info[0]['array-location'])
        if block_info is None or len(block_info) == 0:
            return None
        location = block_info[0]["array-location"]
        frame = location[0][0]
        try:
            segments_at_frame = self.df_tracks.loc[frame]
        except KeyError:
            return np.zeros_like(block, dtype=np.uint8)

        finalized_labels_at_frame = (
            segments_at_frame[
                segments_at_frame["track_id"].isin(self.finalized_track_ids)
            ]
            .index.get_level_values("label")
            .to_list()
        )

        candidate_labels_at_frame = (
            segments_at_frame[
                segments_at_frame["track_id"].isin(self.candidate_track_ids)
            ]
            .index.get_level_values("label")
            .to_list()
        )

        input_vals = finalized_labels_at_frame + candidate_labels_at_frame
        output_vals = [1] * len(finalized_labels_at_frame) + [2] * len(
            candidate_labels_at_frame
        )

        return map_array(block, input_vals, output_vals)

    # transition functions #
    @log_error
    def select_track(self, frame, val, track_id):
        self.track_id = track_id
        segment_labels = np.ones(self.sizeT, dtype=np.uint32) * NOSEL_VALUE
        df = self.df_tracks[self.df_tracks["track_id"] == track_id]
        frames = df.index.get_level_values("frame").values
        labels = df.index.get_level_values("label").values
        segment_labels[frames] = labels

        self.label_edited = np.zeros(len(segment_labels), dtype=bool)
        self.segment_labels = segment_labels
        self.original_segment_labels = segment_labels.copy()
        # used to rewrite track on exit

        row = self.df_divisions[self.df_divisions["parent_track_id"] == track_id]
        print("segment id:", track_id)
        print(segment_labels)
        print(row)
        if len(row) == 1:
            self.frame_childs = list(row.iloc[0][["frame_child1", "frame_child2"]])
            self.label_childs = list(row.iloc[0][["label_child1", "label_child2"]])
        elif len(row) == 0:
            self.frame_childs = []
            self.label_childs = []
        else:
            return
        print(self.frame_childs, self.label_childs)
        self.sel_label_layer.data = [
            d.map_blocks(self.__label_to_selected_label_image, dtype=np.uint8)
            for d in self.label_layer.data
        ]

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

    @log_error
    def switch_track_enter_valid(self):
        iT = self.viewer.dims.current_step[0]
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
            logger.info("switch valid")
            return True

    @log_error
    def switch_track(self, frame, val, track_id):
        direction = choose_direction_by_mbox(self.viewer)

        if not direction:
            return
        elif direction == "forward":
            print("forward ... ")
            df = self.df_tracks[
                (self.df_tracks["track_id"] == track_id)
                & (self.df_tracks.index.get_level_values("frame") >= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values

            self.segment_labels[frame:] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[frame:] = False
            # FIXME revert layer to original
            row = self.df_divisions[self.df_divisions["parent_track_id"] == track_id]

            if len(row) == 1:
                self.frame_childs = row.iloc[0][["frame_child1", "frame_child2"]]
                self.label_childs = row.iloc[0][["label_child1", "label_child2"]]
            elif len(row) == 0:
                self.frame_childs = []
                self.label_childs = []
            self.termination_annotation = ""

        elif direction == "backward":
            df = self.df_tracks[
                (self.df_tracks["track_id"] == track_id)
                & (self.df_tracks.index.get_level_values("frame") <= frame)
            ]
            frames = df.index.get_level_values("frame").values
            labels = df.index.get_level_values("label").values
            self.segment_labels[:frame] = NOSEL_VALUE
            self.segment_labels[frames] = labels
            self.label_edited[:frame] = False

    @log_error
    def daughter_choose_mode_enter_valid(self):
        logger.info("enter daughter choose")
        iT = self.viewer.dims.current_step[0]
        if not iT in self.target_Ts:
            logger.info("this frame is not in target_Ts")
            return False
        previous_iT = self.target_Ts[max(0, self.target_Ts.index(iT) - 1)]
        if (
            self.segment_labels[iT] == NOSEL_VALUE
            and self.segment_labels[previous_iT] == NOSEL_VALUE
        ):
            logger.info("track does not exist in connected timeframe")
            return False
        logger.info("mark division...")
        self.frame_child_candidate = iT
        self.label_child_candidates = []
        return True

    @log_error
    def on_enter_DAUGHTER_CHOOSE_MODE(self, *_):
        logger.info("candidates count: %i", len(self.label_child_candidates))
        if len(self.label_child_candidates) == 2:
            self.finalize_daughter()
            self.to_LABEL_SELECTED()
        else:
            method = choose_division_by_mbox(self.viewer)
            logger.info("%s selected", method)
            if method == "select":
                self.to_DAUGHTER_SWITCH()
            elif method == "draw":
                self.refresh_redraw_label_layer()
                self.to_DAUGHTER_DRAW()
            else:
                self.to_LABEL_SELECTED()

    @log_error
    def daughter_select(self, frame, val, track_id):
        if frame == self.frame_child_candidate:
            self.label_child_candidates.append(int(val))
        else:
            logger.info("frame not correct")

    @log_error
    def daughter_draw_finish(self):
        self.label_child_candidates.append(self.redraw_label_layer.data == 1)

    @log_error
    def finalize_daughter(self):
        assert len(self.label_child_candidates) == 2
        self.frame_childs = []
        self.label_childs = []
        for j, candidate in enumerate(self.label_child_candidates):
            self.label_childs.append(candidate)
            self.frame_childs.append(self.frame_child_candidate)
        self.segment_labels[self.frame_child_candidate :] = NOSEL_VALUE

    @log_error
    def mark_termination_enter_valid(self):
        iT = self.viewer.dims.current_step[0]
        if not np.any(self.sel_label_layer.data[iT] == 1):
            logger.info("track does not exist in connected timeframe")
            return False
        else:
            logger.info("marking termination valid")
            return True

    @log_error
    def mark_termination(self):
        iT = self.viewer.dims.current_step[0]
        termination_annotation, res = get_annotation_of_track_end(
            self.viewer, self.termination_annotations.get(self.track_id, "")
        )
        if res:
            logger.info("marking termination: {termination_annotation}")
            self.termination_annotation = termination_annotation
            if iT < self.segment_labels.shape[0] - 1:
                self.segment_labels[iT + 1 :] = NOSEL_VALUE
        else:
            logger.info("marking termination cancelled")

    @log_error
    def finalize_track(self):
        track_id = self.track_id
        segment_labels = self.segment_labels

        frame_childs = self.frame_childs.copy()
        label_childs = self.label_childs.copy()

        segment_graph = nx.Graph()
        frame_labels = list(enumerate(segment_labels)) + list(
            zip(frame_childs, label_childs)
        )
        relevant_track_ids = np.unique(
            [
                self.df_tracks.loc[(frame, label), "track_id"]
                for frame, label in frame_labels
                if np.isscalar(label)
                and label != NOSEL_VALUE
                and label != NEW_LABEL_VALUE
            ]
        )

        last_frames = {}
        for relevant_track_id in relevant_track_ids:
            df = self.df_tracks[self.df_tracks["track_id"] == relevant_track_id]
            if len(df) == 0:
                continue
            df = df.sort_index(level="frame")
            last_frames[relevant_track_id] = df.index.get_level_values("frame")[-1]
            if len(df) == 1:
                frame, label = df.index[0]
                segment_graph.add_node((frame, label))
            else:
                for ((frame1, label1), _), ((frame2, label2), _) in zip(
                    df.iloc[:-1].iterrows(), df.iloc[1:].iterrows()
                ):
                    segment_graph.add_edge((frame1, label1), (frame2, label2))

        for frame, label in enumerate(segment_labels):
            if label in (NOSEL_VALUE, NEW_LABEL_VALUE):
                continue
            segment_graph.remove_node((frame, label))
            self.df_tracks.loc[(frame, label), "track_id"] = track_id

        for frame, label in zip(frame_childs, label_childs):
            if not np.isscalar(label):
                continue
            neighbors = segment_graph.neighbors((frame, label))
            ancestors = [n for n in neighbors if n[0] < frame]
            if len(ancestors) == 0:
                continue
            else:
                assert len(ancestors) == 1
                ancestor = ancestors[0]
                segment_graph.remove_edge((frame, label), ancestor)

        # relavel divided tracks
        for subsegment in nx.connected_components(segment_graph):
            frame_labels = sorted(subsegment, key=lambda x: x[0])
            original_track_id = self.df_tracks.loc[frame_labels, "track_id"]
            assert np.all(original_track_id.iloc[0] == original_track_id)
            original_track_id = original_track_id.iloc[0]
            last_frame = last_frames[original_track_id]
            frames, _ = zip(*frame_labels)

            self.df_tracks.loc[frame_labels, "track_id"] = self.new_track_id
            if np.any(frames == last_frame):
                ind = self.df_divisions["parent_track_id"] == original_track_id
                if np.any(ind):
                    assert np.sum(ind) == 1
                    self.df_divisions.loc[ind, "parent_track_id"] = self.new_track_id
            self.new_track_id += 1

        def __draw_label(label_image, frame, label):
            # XXX tenative imprementation, faster if directly edit the zarr?
            __dask_compute = (
                lambda arr: arr.compute() if isinstance(arr, da.Array) else arr
            )
            inds = [__dask_compute(i) for i in np.where(label_image)]
            bboxes = [(np.min(ind), np.max(ind) + 1) for ind in inds]
            subimg = np.array(
                self.label_layer.data[frame, 0, 0, slice(*bboxes[0]), slice(*bboxes[1])]
            )
            subimg[tuple((ind - bbox[0]) for ind, bbox in zip(inds, bboxes))] = label
            self.label_layer.data[
                frame, 0, 0, slice(*bboxes[0]), slice(*bboxes[1])
            ] = subimg
            return bboxes

        for redrawn_frame in np.where(self.label_edited)[0]:
            label = self.segment_labels[redrawn_frame]
            if not label in [NOSEL_VALUE, NEW_LABEL_VALUE]:
                __draw_label(
                    self.label_layer.data[redrawn_frame, 0, 0] == label,
                    redrawn_frame,
                    0,
                )
            else:
                label = self.new_label_value

                # FIXME: rewrite with concat
                self.df_tracks = self.df_tracks.append(
                    pd.Series({"track_id": track_id}, name=(redrawn_frame, label))
                )
                self.new_label_value += 1

            bboxes = __draw_label(
                self.sel_label_layer.data[redrawn_frame, 0, 0] == 1,
                redrawn_frame,
                label,
            )
            # set bounding box
            self.df_tracks.loc[(redrawn_frame, label), "bbox_y0"] = bboxes[0][0]
            self.df_tracks.loc[(redrawn_frame, label), "bbox_y1"] = bboxes[0][1]
            self.df_tracks.loc[(redrawn_frame, label), "bbox_x0"] = bboxes[1][0]
            self.df_tracks.loc[(redrawn_frame, label), "bbox_x1"] = bboxes[1][1]

        ind = self.df_divisions["parent_track_id"] == track_id
        if np.any(ind):
            assert np.sum(ind) == 1
            self.df_divisions = self.df_divisions[~ind]
            self.new_track_id += 1

        if len(frame_childs) > 0 and len(label_childs) > 0:
            assert len(frame_childs) == 2 and len(label_childs) == 2
            division_row = {"parent_track_id": track_id}
            for j, (frame_child, label_child) in enumerate(
                zip(frame_childs, label_childs)
            ):
                division_row[f"frame_child{j+1}"] = frame_child
                if np.isscalar(label_child):
                    # means the daughter was selected
                    division_row[f"label_child{j+1}"] = label_child
                    track_id_child = self.df_tracks.loc[
                        (frame_child, label_child), "track_id"
                    ]
                else:
                    bboxes = __draw_label(
                        label_child[0], frame_child, self.new_label_value
                    )
                    division_row[f"label_child{j+1}"] = self.new_label_value
                    # FIXME: rewrite with concat
                    self.df_tracks = self.df_tracks.append(
                        pd.Series(
                            {
                                "track_id": self.new_track_id,
                                "bbox_y0": bboxes[0][0],
                                "bbox_y1": bboxes[0][1],
                                "bbox_x0": bboxes[1][0],
                                "bbox_x1": bboxes[1][1],
                            },
                            name=(frame_child, self.new_label_value),
                        )
                    )
                    track_id_child = self.new_track_id
                    self.new_track_id += 1
                    self.new_label_value += 1
                if not track_id_child in self.finalized_track_ids:
                    logger.info(f"candidate adding ... {track_id_child}")
                    self.candidate_track_ids.add(track_id_child)
            # FIXME: rewrite with concat
            self.df_divisions = self.df_divisions.append(
                division_row, ignore_index=True
            )

        self.finalized_track_ids.add(track_id)
        self.candidate_track_ids.discard(track_id)
        self.termination_annotations[track_id] = self.termination_annotation

        self.finalized_label_layer.data = self.label_layer.data.map_blocks(
            self.__label_to_finalized_label, dtype=np.uint8
        )

    @log_error
    def save_results(self, zarr_path, label_dataset_name, chunks, persist):
        logger.info("saving validation results...")

        if not label_dataset_name.endswith(".travali"):
            label_dataset_name += ".travali"
        zarr_file = zarr.open(zarr_path, "a")
        if label_dataset_name in zarr_file["labels"].keys():
            if_overwrite = ask_ok_or_not(
                self.viewer, "Validation file already exists. Overwrite?"
            )
            if not if_overwrite:
                logger.warning("label not saved")
                return
        logger.info("saving label ...")

        # to avoid IO from/to the same array, save to a temp array and then rename
        label_group = zarr_file["labels"]
        label_chunks = [chunks[0], *chunks[2:]]
        label_data = self.label_layer.data[:, 0, :, :, :].rechunk(label_chunks)
        ds = label_group.create_dataset(
            f"{label_dataset_name}_tmp",
            shape=label_data.shape,
            dtype=label_data.dtype,
            chunks=label_chunks,
            overwrite=True,
        )
        label_data.to_zarr(ds, overwrite=True)
        if label_dataset_name in label_group.keys():
            del label_group[label_dataset_name]
        label_group.store.rename(ds.name, f"{label_group.name}/{label_dataset_name}")
        label_group[label_dataset_name].attrs["target_Ts"] = list(
            map(int, self.target_Ts)
        )

        logger.info("saving segments...")

        segments_group = zarr_file["df_tracks"]
        if label_dataset_name in segments_group.keys():
            del segments_group[label_dataset_name]
        segments_ds = segments_group.create_dataset(
            label_dataset_name,
            data=self.df_tracks.reset_index()[DF_TRACKS_COLUMNS].astype(int).values,
        )
        segments_ds.attrs["finalized_track_ids"] = list(
            map(int, self.finalized_track_ids)
        )
        segments_ds.attrs["candidate_track_ids"] = list(
            map(int, self.candidate_track_ids)
        )
        segments_ds.attrs["termination_annotations"] = {
            int(k): str(v) for k, v in self.termination_annotations.items()
        }

        logger.info("saving divisions...")

        divisions_group = zarr_file["df_divisions"]
        if label_dataset_name in divisions_group.keys():
            del divisions_group[label_dataset_name]
        divisions_group.create_dataset(
            label_dataset_name,
            data=self.df_divisions.reset_index()[DF_DIVISIONS_COLUMNS]
            .astype(int)
            .values,
        )
        logger.info("reading data ...")
        self.label_layer.data = da.from_zarr(label_group[label_dataset_name])[
            :, np.newaxis, :, :, :
        ]
        if persist:
            self.label_layer.data = self.label_layer.data.persist()
        logger.info("saving validation results finished")
