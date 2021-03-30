import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys


def findArucoMarker(img, marker_size: int = 6, total_markers: int = 250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    setting = ''.join(['DICT_', str(marker_size), 'X',
                       str(marker_size), '_', str(total_markers)])
    key = getattr(aruco, setting)
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_param)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def augmentImage(bbox, bbox_id, img, img_aug, draw_id: bool = False):
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_left = bbox[0][2][0], bbox[0][2][1]
    bottom_right = bbox[0][3][0], bbox[0][3][1]

    h, w = img_aug.shape[:2]

    # warp prespective
    pts_dest = np.array([top_left, top_right, bottom_left, bottom_right])
    pts_origin = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    mtx, _ = cv2.findHomography(pts_origin, pts_dest)
    warp = cv2.warpPerspective(img_aug, mtx, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, pts_dest.astype(int), (0, 0, 0))
    img_out = img + warp

    if draw_id:
        cv2.putText(img_out, str(bbox_id), top_left,
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return img_out
