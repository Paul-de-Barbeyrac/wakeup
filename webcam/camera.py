import cv2
import numpy as np
from tensorflow import keras

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_righteye_2splits.xml')

model_deep = keras.models.load_model("model/model.h5")


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        tickmark = cv2.getTickCount()
        success, image = self.video.read()
        lpred_output = "Unknown"
        rpred_output = "Unknown"
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

            left_eye = left_eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.4,
                minNeighbors=10,
                minSize=(60, 60)
            )

            right_eye = right_eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.4,
                minNeighbors=10,
                minSize=(60, 60)
            )

            for (ex, ey, ew, eh) in left_eye:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

                l_eye = image[y:y + h, x:x + w]
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (28, 28))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(28, 28, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                lpred = model_deep.predict_classes(l_eye)
                if round(lpred[0][0]) == 1:
                    lpred_output = 'Open'
                    break
                if round(lpred[0][0]) == 0:
                    lpred_output = 'Closed'
                    break

            for (ex, ey, ew, eh) in right_eye:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                r_eye = image[y:y + h, x:x + w]
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (28, 28))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(28, 28, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                rpred = model_deep.predict_classes(r_eye)
                if round(rpred[0][0]) == 1:
                    rpred_output = 'Open'
                    break
                if round(rpred[0][0]) == 0:
                    rpred_output = 'Closed'
                    break


        frame_flip = cv2.flip(image, 1)  # flip vertically
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
        cv2.putText(frame_flip, f"FPS: {round(fps, 1)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame_flip, f"LEFT : {lpred_output}", (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        cv2.putText(frame_flip, f"RIGHT : {rpred_output}", (80, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)

        return jpeg.tobytes()
