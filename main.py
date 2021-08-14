import cv2


vid = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,
    varThreshold=40
    )

while(True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    #print(height, width)
    roi = frame[0:720, 0:640]
    
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # cv2.circle(frame, (50, 50), 30, (0, 0, 255), -1)
    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



vid.release()
cv2.destroyAllWindows()
