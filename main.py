# Sounds Without Strings

import time

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
from numpy.random.mtrand import f

MODEL_PATH = "model/gesture_recognizer.task"

mido.set_backend("mido.backends.rtmidi")

try:
    midi_out = mido.open_output("IAC Driver Python MIDI 1")  # type: ignore
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
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[*] Empty camera frame")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms = int(time.time() * 1000)

        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        cv2.imshow("Sounds Without Strings", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
