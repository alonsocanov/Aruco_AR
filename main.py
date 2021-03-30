import cv2
import aruco_marker
import utils
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Aruco AR')
    parser.add_argument('--path', type=str, default='images/display/sponge_bob.gif',
                        help='path to image or video to superpose')
    parser.add_argument('--device', type=int, default=0,
                        help='camera device to use')
    parser.add_argument('--pose', type=int, default=0,
                        help='pose axis')
    parser.add_argument('--cube', type=int, default=0,
                        help='cube')
    args = parser.parse_args()

    cap = utils.checkDevice(args.device)

    mtx = utils.loadData('logi_720/new_camera_matrix.txt')
    dst = utils.loadData('logi_720/camera_distortion.txt')

    utils.checkFile(args.path)

    vid_ext = ['.gif', '.mp4', '.avi', '.mov']

    flag_img = args.path.endswith('.jpg') or args.path.endswith('.png')
    flag_vid = False

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))

    # size = (frame_width, frame_height)

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('data/aug_cube.avi', fourcc, 20.0, size)

    if not flag_img:
        for ext in vid_ext:
            if args.path.endswith(ext):
                flag_vid = True
                break

    if flag_img:
        img_aug = cv2.imread(args.path)
    elif flag_vid:
        aug = cv2.VideoCapture(args.path)
    else:
        message = ' '.join(['File not allowed:', args.path])
        print(message)
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        aruco_found = aruco_marker.findArucoMarker(frame)
        if len(aruco_found[0]) != 0:
            if not args.pose and not args.cube:
                if flag_vid:
                    ret, img_aug = aug.read()
                    if not ret:
                        aug = cv2.VideoCapture(args.path)
                        ret, img_aug = aug.read()
                for bbox, bbox_id in zip(aruco_found[0], aruco_found[1]):
                    frame = aruco_marker.augmentImage(
                        bbox, bbox_id, frame, img_aug, True)
            elif args.pose:
                rvecs, tvecs, obj_pts = cv2.aruco.estimatePoseSingleMarkers(
                    aruco_found[0], 0.5, mtx, dst)
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.aruco.drawAxis(frame, mtx, dst, rvec, tvec, .5)
            elif args.cube:
                rvecs, tvecs, obj_pts = cv2.aruco.estimatePoseSingleMarkers(
                    aruco_found[0], 0.5, mtx, dst)
                markerLength = .5
                m = markerLength/2
                pts = np.float32([[-m, m, m], [-m, -m, m], [m, -m, m],
                                  [m, m, m], [-m, m, 0], [-m, -m, 0],
                                  [m, -m, 0], [m, m, 0]])
                for rvec, tvec in zip(rvecs, tvecs):
                    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, mtx, dst)
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    frame = cv2.drawContours(
                        frame, [imgpts[:4]], -1, (0, 0, 255), 4)
                    for i, j in zip(range(4), range(4, 8)):
                        frame = cv2.line(frame, tuple(imgpts[i]), tuple(
                            imgpts[j]), (0, 0, 255), 4)
                        frame = cv2.drawContours(
                            frame, [imgpts[4:]], -1, (0, 0, 255), 4)

        # cv2.imwrite('data/static.jpg', frame)
        cv2.imshow('Image', frame)
        # out.write(frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    # out.release()
    if flag_vid:
        aug.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
