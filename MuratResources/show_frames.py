"""Read two video clips from files, and show the frames on a window.

This module reads two video clips from files, loads them into the memory and
shows frames of each video.

Arguments:
    left: The path of the first video (i.e. usually the left camera video)
    right: The path of the second video (i.e. usually the right camera video)
    width: Width of the window in pixels
    height: Height of the window in pixels

Example Usage: python3 video_sync.py left.MP4 right.MP4
"""
import cv2
import numpy as np
import argparse
import os


def nothing(x):
    """Do nothing used as a callback function in cv2.createTrackbar()."""
    pass


def find_frame(x, video):
    """Get xth frame of the video."""
    video.set(cv2.CAP_PROP_POS_FRAMES, x)
    retval, image = video.read()
    
    if retval:
        return image
    else:
        return np.zeros((720, 1280, 3), dtype=np.uint8)


def show(settings):
    """."""
    # Create a window to show the frames and the trackbars
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', settings["height"], settings["width"])
    # check if the video files exit, if not raise necessary errors.
    if not os.path.exists(settings["left"]):
        raise IOError("Left video does not exist!")

    if not os.path.exists(settings["right"]):
        raise IOError("Right video does not exist!")

    # open the videos
    l_video = cv2.VideoCapture(settings["left"])
    r_video = cv2.VideoCapture(settings["right"])
    # count how many frames there are in the videos
    l_nframes = int(l_video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    r_nframes = int(r_video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    # create trackbars
    cv2.createTrackbar('Left', 'image', 0, l_nframes, nothing)
    cv2.createTrackbar('Right', 'image', 0, r_nframes, nothing)
    # the last position of the trackbar values
    l_pos_last = None
    r_pos_last = None

    while(1):
        # get current positions of the trackbars
        l_pos = cv2.getTrackbarPos('Left', 'image')
        r_pos = cv2.getTrackbarPos('Right', 'image')
        # if the trackbar positions have changed, get the corresponding frames
        if l_pos != l_pos_last:
            frame_l = find_frame(l_pos, l_video)

        if r_pos != r_pos_last:
            frame_r = find_frame(r_pos, r_video)
        # if either of the trackbar value has changed, update the canvas
        if l_pos != l_pos_last or r_pos != r_pos_last:
            img = np.concatenate([frame_l, frame_r], axis=1)

        l_pos_last = l_pos
        r_pos_last = r_pos

        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # initialize the argument parser
    parser = argparse.ArgumentParser()
    # define the arguments needed to run this python module
    parser.add_argument("left",
                        help="path of the video obtained by the left camera")
    parser.add_argument("right",
                        help="path of the video obtained by the right camera")
    parser.add_argument("--width", default=1500, type=int,
                        help="width of the window in pixels")
    parser.add_argument("--height", default=900, type=int,
                        help="height of the the window in pixels")
    # parse arguments
    args = parser.parse_args()
    # assign all the args, into a dict
    settings = {"left": args.left,
                "right": args.right,
                "width": args.width,
                "height": args.height}
    # run the whole show
    show(settings)
