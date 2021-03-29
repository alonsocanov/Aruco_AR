import cv2
import aruco_marker
import sys


def main():
    cap = aruco_marker.checkDevice(0)

    file_path = 'images/display/sponge_bob.gif'

    if not aruco_marker.checkFile(file_path):
        message = ' '.join(['File does not exist:', file_path])
        print(message)
        sys.exit(1)

    flag_img = file_path.endswith('.jpg') or file_path.endswith('.png')
    flag_vid = file_path.endswith('.gif') or file_path.endswith(
        '.mov') or file_path.endswith('.mp4') or file_path.endswith('.avi')

    if flag_img:
        img_aug = cv2.imread(file_path)
    elif flag_vid:
        aug = cv2.VideoCapture(file_path)
    else:
        message = ' '.join(['File not allowed:', file_path])
        print(message)
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        aruco_found = aruco_marker.findArucoMarker(frame)
        if len(aruco_found[0]) != 0:
            if flag_vid:
                ret, img_aug = aug.read()
                if not ret:
                    aug = cv2.VideoCapture(file_path)
                    ret, img_aug = aug.read()

            for bbox, bbox_id in zip(aruco_found[0], aruco_found[1]):
                frame = aruco_marker.augmentImage(
                    bbox, bbox_id, frame, img_aug, True)

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
