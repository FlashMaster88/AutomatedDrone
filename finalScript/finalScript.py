import argparse
import olympe
import os
import sys
import time
from olympe.video.pdraw import Pdraw, PdrawState
from olympe.video.renderer import PdrawRenderer
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from olympe.messages.ardrone3.Piloting import TakeOff, moveBy, Landing
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
import math
import cv2

from Detector import *


# DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1") # Real Drone
DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1") # Simulation Drone
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")
detector = Detector(use_cuda = False)

drone = olympe.Drone(DRONE_IP)

def yuv_frame_cb(yuv_frame):
    """
    This function will be called by Olympe for each decoded YUV frame.

        :type yuv_frame: olympe.VideoFrame
    """

    # the VideoFrame.info() dictionary contains some useful information
    # such as the video resolution
    info = yuv_frame.info()
    height, width = (  # noqa
        info["raw"]["frame"]["info"]["height"],
        info["raw"]["frame"]["info"]["width"],
    )

    # yuv_frame.vmeta() returns a dictionary that contains additional
    # metadata from the drone (GPS coordinates, battery percentage, ...)

    # convert pdraw YUV flag to OpenCV YUV flag
    cv2_cvt_color_flag = {
        olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
        olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
    }[yuv_frame.format()]
    # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
    # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

    # Use OpenCV to convert the yuv frame to RGB
    cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)  # noqa

    upper_left_x, upper_left_y, lower_right_x, lower_right_y = detector.processImage(cv2frame)
    # print(upper_left_x, upper_left_y, lower_right_x, lower_right_y)
    height = lower_right_y - upper_left_y

    #moveBy(front-back, right-left, turn-360, 0, _timeout=seconds)
    #moveBy(-100 to +100, -100 to +100, turn-360, 0, _timeout=seconds)

    print("upper_left_x: " + str(upper_left_x))
    print("height: " + str(height))

    front_back_speed = int(0)
    left_right_speed = int(0)

    if (upper_left_x == 0 and upper_left_y == 0):
        front_back_speed = int(0)
        left_right_speed = int(0)

    if (upper_left_x < 600 and upper_left_x != 0):
        left_right_speed = int(-1)

    elif (upper_left_x == 600):
        left_right_speed = int(0)

    elif (upper_left_x > 600):
        left_right_speed = int(1)


    if (height < 400 and height != 0):
        front_back_speed = int(1)

    elif (height == 400):
        front_back_speed = int(0)

    elif (height > 400):
        front_back_speed = int(-1)
        
    # drone(moveBy(0, 100, 0, 0, _timeout=0.01)).wait().success()
    # drone(moveBy(front_back_speed, left_right_speed, 0, 0, _timeout = movement_time)).success()

    print("front_back_speed: " + str(front_back_speed))
    print("left_right_speed: " + str(left_right_speed))

    # assert drone(moveBy(front_back_speed, left_right_speed, 0, 0).wait().success() 
    # >> FlyingStateChanged(state="hovering", _timeout=5)).success()

    #THIS WORKS #####################################################
    assert drone(

        moveBy(front_back_speed, left_right_speed, 0, 0)

        >> FlyingStateChanged(state="hovering", _timeout=5)

    ).wait(0.05).success()
    ##################################################################



    #     >> FlyingStateChanged(state="hovering", _timeout=5)

    # ).wait().success()



    # drone(moveBy(0, 2, 0, 0, _timeout=5)).wait().success()

    # drone(moveBy(front_back_speed, left_right_speed, 0, 0)).success() # This is the movement command

def main(argv):
    
    # Connect the the drone
    # drone = olympe.Drone(DRONE_IP)
    assert drone.connect(retry=3)

    # assert drone(TakeOff()).wait().success()

    assert drone(TakeOff()>> FlyingStateChanged(state="hovering", _timeout=5)).wait().success()

    # drone(moveBy(0, -2, 0, 0, _timeout=1)).wait().success()

    # drone(moveBy(0, -2, 0, 0)).wait().success()
    
    parser = argparse.ArgumentParser(description="Olympe Pdraw Example")
    parser.add_argument(
        "-u",
        "--url",
        default=f"rtsp://{DRONE_IP}:{DRONE_RTSP_PORT}/live",
        help=(
            "Media resource (rtsp:// or file://) URL.\n"
            "See olympe.Pdraw.play documentation"
        ),
    )
    parser.add_argument("-m", "--media-name", default="DefaultVideo")
    args = parser.parse_args(argv)
    pdraw = Pdraw()

    # Uncomment the following line, to test this OpenCV frame processing callback function
    # This function requires `pip3 install opencv-python`.

    pdraw.set_callbacks(raw_cb=yuv_frame_cb)
    pdraw.play(url=args.url, media_name=args.media_name)
    renderer = PdrawRenderer(pdraw=pdraw)
    assert pdraw.wait(PdrawState.Playing, timeout=50)
    if args.url.endswith("/live"):
        # Let's see the live video streaming for 10 seconds
        time.sleep(45) # This is the line that dictates timing (seconds) 
        pdraw.close()
        pdraw.dispose()
        assert drone(Landing()).wait().success()
        assert drone.disconnect()
        timeout = 50
    else:
        # When replaying a video, the pdraw stream will be closed automatically
        # at the end of the video
        # For this is example, this is the replayed video maximal duration:
        timeout = 900
    assert pdraw.wait(PdrawState.Closed, timeout=timeout)
    renderer.stop()
    pdraw.dispose()

    assert drone(Landing()).wait().success()
    assert drone.disconnect()


def test_pdraw():
    main([])


if __name__ == "__main__":
    main(sys.argv[1:])
