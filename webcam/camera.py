import time

import cv2
import dlib
import numpy as np
from tensorflow import keras

# face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
# left_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_lefteye_2splits.xml')
# right_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_righteye_2splits.xml')

model_deep = keras.models.load_model("model/model.h5")


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


class VideoCamera(object):
    def __init__(self, sp, detector, sleeping_counter=0):
        self.video = cv2.VideoCapture(0)
        self.sp = sp
        self.detector = detector
        self.sleeping_counter = sleeping_counter

    def __del__(self):
        self.video.release()

    def get_frame(self):
        start1 = time.time()
        tickmark = cv2.getTickCount()
        success, image = self.video.read()

        if success:
            # print(image.shape)
            image = rescale_frame(image, percent=50)
            # print(image.shape)
            # print("step 1:" + str(time.time() - start1))
            # Convert to dlib
            start2 = time.time()
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print("step 2:" + str(time.time() - start2))
            # dlib face detection

            start3 = time.time()
            detector = self.detector
            # print("step 3:" + str(time.time() - start3))

            # OPTIMISER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            start4 = time.time()
            detections = detector(img, 1)
            # print("step 4:" + str(time.time() - start4))

            # print(f"{start1 - start}")
            # Find landmarks
            start5 = time.time()
            sp = self.sp
            faces = dlib.full_object_detections()
            # print("step 5:" + str(time.time() - start5))
            start6 = time.time()
            # print("step 6:" + str(time.time() - start6))

            start7 = time.time()
            for det in detections:
                faces.append(sp(img, det))
            # print("step 7:" + str(time.time() - start7))

            start8 = time.time()

            # Bounding box and eyes
            bb = [i.rect for i in faces]
            bb = [((i.left(), i.top()),
                   (i.right(), i.bottom())) for i in bb]  # Convert out of dlib format

            right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
            right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]  # Convert out of dlib format

            left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
            left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]  # Convert out of dlib format

            # Display
            imgd = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to OpenCV

            proba_left = 0
            proba_right = 0

            for eye in left_eyes:
                width = max(eye, key=lambda x: x[0])[0] - min(eye, key=lambda x: x[0])[0]
                center_x = round((min(eye, key=lambda x: x[0])[0] + max(eye, key=lambda x: x[0])[0]) / 2)
                center_y = round((min(eye, key=lambda x: x[1])[1] + max(eye, key=lambda x: x[1])[1]) / 2)

                # Plot center of eye
                # cv2.circle(imgd, (center_x, center_y), 2, (0, 255, 0), -1)
                extract_eye_left = imgd[center_y - width:center_y + width, center_x - width:center_x + width]
                extract_eye_left = cv2.cvtColor(extract_eye_left, cv2.COLOR_BGR2GRAY)
                extract_eye_left = cv2.resize(extract_eye_left, (28, 28))

                # Save image
                # cv2.imwrite(f"left2-eye-{center_x}-{center_y}.jpg", extract_eye_left)

                extract_eye_left = extract_eye_left / 255
                extract_eye_left = extract_eye_left.reshape(28, 28, -1)
                extract_eye_left = np.expand_dims(extract_eye_left, axis=0)
                pred = model_deep.predict(extract_eye_left)
                proba_left = pred[0][0]
                if proba_left > 0.9:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                # print(round(proba, 4))
                cv2.rectangle(imgd, (center_x - width, center_y - width), (center_x + width, center_y + width), color,
                              0)

                text_to_display = f"{round(float(proba_left), 2)}"
                coordinates = (center_x - width, center_y - width - 5)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.25
                thickness = 0
                cv2.putText(imgd, text_to_display, coordinates, font, font_scale, color, thickness)

            for eye in right_eyes:
                width = max(eye, key=lambda x: x[0])[0] - min(eye, key=lambda x: x[0])[0]
                center_x = round((min(eye, key=lambda x: x[0])[0] + max(eye, key=lambda x: x[0])[0]) / 2)
                center_y = round((min(eye, key=lambda x: x[1])[1] + max(eye, key=lambda x: x[1])[1]) / 2)

                # Plot center of eye
                # cv2.circle(imgd, (center_x, center_y), 2, (0, 255, 0), -1)
                extract_eye_right = imgd[center_y - width:center_y + width, center_x - width:center_x + width]
                extract_eye_right = cv2.cvtColor(extract_eye_right, cv2.COLOR_BGR2GRAY)
                extract_eye_right = cv2.resize(extract_eye_right, (28, 28))

                # Save image
                # cv2.imwrite(f"right-eye-{center_x}-{center_y}.jpg", extract_eye_right)

                extract_eye_right = extract_eye_right / 255
                extract_eye_right = extract_eye_right.reshape(28, 28, -1)
                extract_eye_right = np.expand_dims(extract_eye_right, axis=0)
                pred = model_deep.predict(extract_eye_right)
                proba_right = pred[0][0]
                if proba_right > 0.9:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                # print(round(proba_right, 4))
                cv2.rectangle(imgd, (center_x - width, center_y - width), (center_x + width, center_y + width), color,
                              0)

                text_to_display = f"{round(float(proba_right), 2)}"
                coordinates = (center_x - width, center_y - width - 5)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.25
                thickness = 0
                cv2.putText(imgd, text_to_display, coordinates, font, font_scale, color, thickness)

            if faces:
                if round((proba_left + proba_right) / 2) == 1:
                    if self.sleeping_counter > 0:
                        self.sleeping_counter = self.sleeping_counter - 1
                elif round((proba_left + proba_right) / 2) == 0:
                    self.sleeping_counter = self.sleeping_counter + 2

            for i in bb:
                cv2.rectangle(imgd, i[0], i[1], (255, 0, 0), 0)  # Bounding box

            # frame_flip = cv2.flip(imgd, 1)  # flip vertically
            frame_flip = imgd
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
            cv2.putText(frame_flip, f"FPS: {round(fps, 1)}", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 0)
            cv2.putText(frame_flip, f"Drowsiness: {self.sleeping_counter}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 0, 0), 0)
            cv2.rectangle(frame_flip, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), self.sleeping_counter)

            ret, jpeg = cv2.imencode('.jpg', frame_flip)
            # print("step 8:" + str(time.time() - start8))
            # print("---------------------------")
            return jpeg.tobytes()
