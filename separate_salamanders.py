#!/usr/bin/env python3
"""Separate salamanders."""

# pylint: disable=no-member,bad-option-value

import cv2
import numpy as np


INCHES = 32


def find_labels():
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("canny1", "Tracking", 320, 1024, nothing)
    cv2.createTrackbar("canny2", "Tracking", 100, 1024, nothing)
    # cv2.createTrackbar("threshold", "Tracking", 0, 1024, nothing)
    # cv2.createTrackbar("min_line", "Tracking", 0, 1024, nothing)
    # cv2.createTrackbar("max_gap", "Tracking", 0, 1024, nothing)

    while True:
        original = cv2.imread('data/samples/R0009611.JPG')
        # original = cv2.resize(original, (900, 700))
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        canny1 = cv2.getTrackbarPos("canny1", "Tracking")
        canny2 = cv2.getTrackbarPos("canny2", "Tracking")
        # threshold = cv2.getTrackbarPos("threshold", "Tracking")
        # min_line = cv2.getTrackbarPos("min_line", "Tracking")
        # max_gap = cv2.getTrackbarPos("max_gap", "Tracking")

        edges = cv2.Canny(gray, canny1, canny2)

        # lines = cv2.HoughLinesP(
        #     edges, 1, np.pi/360, threshold,
        #     minLineLength=min_line, maxLineGap=max_gap)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imshow("gray", gray)
        cv2.imshow("edges", edges)
        # cv2.imshow("lines", original)

        key = cv2.waitKey(1)
        if key == 27:
            break


def nothing(_):
    pass


def find_salamanders():
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 140, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 100, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    while True:
        frame = cv2.imread('data/samples/R0009611.JPG')
        frame = cv2.resize(frame, (493, 326))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, l_b, u_b)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        key = cv2.waitKey(1)
        if key == 27:
            break

    # lower(0, 80, 0) upper(80, 255, 255)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # find_salamanders()
    find_labels()
