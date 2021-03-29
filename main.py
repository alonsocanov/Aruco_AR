import cv2
import aruco_marker


def main():
    cap = cv2.VideoCapture(0)

    # img_aug = cv2.imread('images/display/CAHIBO.jpg')
    aug = cv2.VideoCapture('images/display/sponge_bob.gif')

    while True:
        ret, frame = cap.read()

        aruco_found = aruco_marker.findArucoMarker(frame)

        if len(aruco_found[0]) != 0:
            ret, img_aug = aug.read()
            for bbox, bbox_id in zip(aruco_found[0], aruco_found[1]):
                frame = aruco_marker.augmentImage(
                    bbox, bbox_id, frame, img_aug, True)
        cv2.imshow('Image', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    aug.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
