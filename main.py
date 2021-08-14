import cv2


vid = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, frame = vid.read()

    mask = object_detector.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    # cv2.circle(frame, (50, 50), 30, (0, 0, 255), -1)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



vid.release()
cv2.destroyAllWindows()
