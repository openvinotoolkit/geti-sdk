# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import shutil
import sys

import cv2
import ffmpeg
import pytest

from geti_sdk.demos import predict_video_on_local


class TestPredictVideo:
    @pytest.mark.parametrize("preserve_audio", [True, False])
    def test_predict_video_on_local(
        self,
        fxt_video_path_dice: str,
        fxt_deployment_path_dice: str,
        preserve_audio: bool,
    ) -> None:
        # Act
        result_filepath = predict_video_on_local(
            video_path=fxt_video_path_dice,
            deployment=fxt_deployment_path_dice,
            preserve_audio=preserve_audio,
        )

        # file extenstion should be preserved.
        assert (
            os.path.splitext(fxt_video_path_dice)[1]
            == os.path.splitext(result_filepath)[1]
        )

        # frame width, height, count, and fps should be preserved.
        cap_src = cv2.VideoCapture(fxt_video_path_dice)
        cap_dst = cv2.VideoCapture(result_filepath)
        assert cap_src.isOpened() and cap_dst.isOpened()
        assert int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH)) == int(
            cap_dst.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        assert int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT)) == int(
            cap_dst.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        assert int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT)) == int(
            cap_dst.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        assert (
            abs(cap_src.get(cv2.CAP_PROP_FPS) - cap_dst.get(cv2.CAP_PROP_FPS))
            < sys.float_info.epsilon
        )
        cap_src.release()
        cap_dst.release()

        if preserve_audio and shutil.which("ffmpeg") is not None:
            # original audio should be preserved.
            probe_src = ffmpeg.probe(fxt_video_path_dice, select_streams="a")
            probe_dst = ffmpeg.probe(result_filepath, select_streams="a")
            if probe_src["streams"]:
                assert probe_dst["streams"]

        # delete output video
        os.remove(result_filepath)

        # Act and assert
        with pytest.raises(ValueError):
            predict_video_on_local(
                video_path=None,
                deployment=fxt_deployment_path_dice,
                preserve_audio=preserve_audio,
            )
        with pytest.raises(ValueError):
            predict_video_on_local(
                video_path=fxt_video_path_dice,
                deployment=None,
                preserve_audio=preserve_audio,
            )
        with pytest.raises(ValueError):
            predict_video_on_local(
                video_path=fxt_video_path_dice,
                deployment=fxt_deployment_path_dice + "/",
                preserve_audio=preserve_audio,
            )
