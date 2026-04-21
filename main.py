# Sounds Without Strings

import cv2
import mediapipe as mp
import mido
import rtmidi
from mediapipe.tasks import python
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
from mediapipe.tasks.python.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureRecognizerOptions,
    GestureRecognizerResult,
)

MODEL_PATH = "/model/gesture_recognizer.task"

mido.set_backend("mido.backends.rtmidi")

try:
    midi_out = mido.open_output("Python MIDI 1")  # type: ignore
except OSError:
    print("Available ports:", mido.Backend().get_output_names())
    print("Please change the port name in the code to match one of the above.")
    exit()


def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    print("gesture recognition result: {}".format(result))


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

with GestureRecognizer.create_from_options(options) as recognizer:
    pass
