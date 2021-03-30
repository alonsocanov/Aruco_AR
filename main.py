import cv2
import aruco_marker
import utils
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Aruco AR')
    parser.add_argument('--path', type=str, default='images/display/sponge_bob.gif',
                        help='path to image or video to superpose')
    parser.add_argument('--device', type=int, default=0,
                        help='camera device to use')
    parser.add_argument('--pose', type=int, default=0,
                        help='pose axis')
    args = parser.parse_args()

    cap = utils.checkDevice(args.device)

    mtx = utils.loadData('logi_720/new_camera_matrix.txt')
    dst = utils.loadData('logi_720/camera_distortion.txt')

    utils.checkFile(args.path)

    flag_img = args.path.endswith('.jpg') or args.path.endswith('.png')
    flag_vid = args.path.endswith('.gif') or args.path.endswith(
        '.mov') or args.path.endswith('.mp4') or args.path.endswith('.avi')

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
            if not args.pose:
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
                    cv2.aruco.drawAxis(frame, mtx, dst, rvec, tvec, .2)

        cv2.imshow('Image', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    if flag_vid:
        aug.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
