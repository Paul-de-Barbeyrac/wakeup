import cv2

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
        minNeighbors=6,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
        minSize=(200, 200)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(60, 60)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
    cv2.putText(img, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
