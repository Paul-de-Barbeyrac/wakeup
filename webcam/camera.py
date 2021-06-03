import cv2

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_eye.xml')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        tickmark = cv2.getTickCount()
        success, image = self.video.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
            minNeighbors=6,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
            minSize=(200, 200)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(60, 60)
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        frame_flip = cv2.flip(image, 1)  # flip vertically
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
        cv2.putText(frame_flip, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)

        return jpeg.tobytes()
