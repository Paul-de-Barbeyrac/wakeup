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

        raw_img = image
        raw_img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        plot_img = raw_img

        faces = face_cascade.detectMultiScale(
            raw_img_gray,
            scaleFactor=1.4,
            # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
            minNeighbors=1,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
            minSize=(200, 200)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
        )

        for (x, y, w, h) in faces:
            # cv2.imwrite('raw_image.jpg', raw_img)
            cv2.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

            face_img = raw_img[y:y + h, x:x + w]
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # cv2.imwrite(str(x) + str(y) + str(w) + str(h) + '_faces.jpg', raw_img)

            left_eye = left_eye_cascade.detectMultiScale(
                face_img_gray,
                scaleFactor=1.4,
                minNeighbors=1,
                minSize=(50, 50)
            )

            right_eye = right_eye_cascade.detectMultiScale(
                face_img_gray,
                scaleFactor=1.4,
                minNeighbors=1,
                minSize=(50, 50)
            )

            for (lx, ly, lw, lh) in left_eye:
                cv2.rectangle(plot_img, (x + lx, y + ly), (x + lx + lw, y + ly + lh), (0, 0, 255), 3)

                l_eye = face_img[ly:ly + lh, lx:lx + lw]
                # cv2.imwrite('left_eye.jpg', l_eye)
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (28, 28))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(28, 28, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                lpred = model_deep.predict_classes(l_eye)
                if round(lpred[0][0]) == 1:
                    lpred_output = 'OPEN'
                    break
                if round(lpred[0][0]) == 0:
                    lpred_output = 'CLOSED'
                    break

            for (rx, ry, rw, rh) in right_eye:
                cv2.rectangle(plot_img, (x + rx, y + ry), (x + rx + rw, y + ry + rh), (0, 255, 0), 3)

                r_eye = face_img[ry:ry + rh, rx:rx + rw]
                # cv2.imwrite('right_eye.jpg', r_eye)
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (28, 28))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(28, 28, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                rpred = model_deep.predict_classes(r_eye)
                if round(rpred[0][0]) == 1:
                    rpred_output = 'OPEN'
                    break
                if round(rpred[0][0]) == 0:
                    rpred_output = 'CLOSED'
                    break

        frame_flip = cv2.flip(plot_img, 1)  # flip vertically
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
        cv2.putText(frame_flip, f"FPS: {round(fps, 1)}", (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame_flip, f"LEFT : {lpred_output}", (0, 60), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255),
                    2)
        cv2.putText(frame_flip, f"RIGHT : {rpred_output}", (0, 90), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0),
                    2)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)

        return jpeg.tobytes()
