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
import math
import cv2

from Detector import *


DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")
detector = Detector(use_cuda = False)

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

    print(type(cv2frame))

    #cv2.imshow('1' , np.array(cv2frame, dtype = np.uint8 ))
    #cv2.imshow("output", cv2frame)
    #plt.imshow(cv2frame, interpolation='nearest')
    #plt.show()
    # img = Image.fromarray(cv2frame, 'RGB')
    # img.save('my.jpg')
    # img.show()
    detector.processImage(cv2frame)

def main(argv):
    
    # Connect the the drone
    drone = olympe.Drone(DRONE_IP)
    assert drone.connect(retry=3)

    #assert drone(TakeOff()).wait().success()
    #drone(moveBy(0, 2, 0, 0, _timeout=1)).wait().success()
    
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
    assert pdraw.wait(PdrawState.Playing, timeout=500)
    if args.url.endswith("/live"):
        # Let's see the live video streaming for 10 seconds
        time.sleep(1000)
        pdraw.close()
        timeout = 500
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
