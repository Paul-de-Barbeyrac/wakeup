import base64
import io

import cv2
import dlib
import numpy as np
from PIL import Image
from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from tensorflow import keras

from webcam.camera import VideoCamera

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')
left_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('opencv/haarcascade_righteye_2splits.xml')
model_deep = keras.models.load_model("model/model.h5")


def readb64(base64_string):
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def index(request):
    return render(request, 'webcam/home.html')


def video(request):
    return render(request, 'webcam/video.html')


def image(request):
    raw_image = None
    processed_image = None
    image_opencv = None
    if request.method == 'POST':
        image = request.FILES['file'].read()
        image_base64 = base64.b64encode(image)
        image_ascii = image_base64.decode('ascii')
        raw_image = f"data:image/jpeg;base64,{image_ascii}"

        # lpred_output = "Unknown"
        # rpred_output = "Unknown"

        raw_img = readb64(image_ascii)

        # Convert to dlib
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # dlib face detection
        detector = dlib.get_frontal_face_detector()
        detections = detector(img, 1)

        # Find landmarks
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = dlib.full_object_detections()
        for det in detections:
            faces.append(sp(img, det))

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
        for i in bb:
            cv2.rectangle(imgd, i[0], i[1], (255, 0, 0), 5)  # Bounding box

        padding = 60
        for eye in right_eyes:
            cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0] - padding, max(eye, key=lambda x: x[1])[1] - padding),
                          (min(eye, key=lambda x: x[0])[0] + padding, min(eye, key=lambda x: x[1])[1] + padding),
                          (0, 0, 255), 3)
            # for point in eye:
            #     cv2.circle(imgd, (point[0], point[1]), 2, (0, 255, 0), -1)

        for eye in left_eyes:
            cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0] - padding, max(eye, key=lambda x: x[1])[1] - padding),
                          (min(eye, key=lambda x: x[0])[0] + padding, min(eye, key=lambda x: x[1])[1] + padding),
                          (0, 0, 255), 3)
            # for point in eye:
            #     cv2.circle(imgd, (point[0], point[1]), 2, (0, 0, 255), -1)

        ret, frame_buff = cv2.imencode('.jpg', imgd)
        image_base64_cv = base64.b64encode(frame_buff)
        image_ascii_cv = image_base64_cv.decode('ascii')
        processed_image = f"data:image/jpeg;base64,{image_ascii_cv}"

        # raw_img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        # plot_img = raw_img
        #
        # faces = face_cascade.detectMultiScale(
        #     raw_img_gray,
        #     scaleFactor=1.3,
        #     # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
        #     minNeighbors=3,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
        #     minSize=(30, 30)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
        # )
        #
        # for (x, y, w, h) in faces:
        #     # cv2.imwrite('raw_image.jpg', raw_img)
        #     face_img = raw_img[y:y + h, x:x + w]
        #     face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        #     # cv2.imwrite(str(x) + str(y) + str(w) + str(h) + '_faces.jpg', raw_img)
        #
        #     left_eye = left_eye_cascade.detectMultiScale(
        #         face_img_gray,
        #         scaleFactor=1.4,
        #         minNeighbors=6,
        #         minSize=(45, 45)
        #     )
        #
        #     right_eye = right_eye_cascade.detectMultiScale(
        #         face_img_gray,
        #         scaleFactor=1.4,
        #         minNeighbors=6,
        #         minSize=(45, 45)
        #     )
        #
        #     print(left_eye)
        #     print(type(left_eye))
        #     print(np.round(np.mean(left_eye, axis=0)).astype(int))
        #
        #     (lx, ly, lw, lh) = np.round(np.mean(left_eye, axis=0)).astype(int)
        #     l_eye = face_img[ly:ly + lh, lx:lx + lw]
        #     cv2.imwrite('left_eye.jpg', l_eye)
        #     l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        #     l_eye = cv2.resize(l_eye, (28, 28))
        #     l_eye = l_eye / 255
        #     l_eye = l_eye.reshape(28, 28, -1)
        #     l_eye = np.expand_dims(l_eye, axis=0)
        #     lpred = model_deep.predict_classes(l_eye)
        #     lpred_proba = lpred[0][0]
        #     if round(lpred_proba) == 1:
        #         text_to_display = f"Eye opening - {round(lpred_proba, 2)}"
        #         coordinates = (x + lx, y + ly)
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 0.4
        #         color_open = (0, 255, 0)
        #         thickness = 3
        #         cv2.rectangle(plot_img, (x + lx, y + ly), (x + lx + lw, y + ly + lh), color_open, thickness)
        #         cv2.putText(plot_img, text_to_display, coordinates, font, font_scale, color_open, thickness)
        #     elif round(lpred_proba) == 0:
        #         text_to_display = f"Eye opening - {round(lpred_proba, 2)}"
        #         coordinates = (x + lx, y + ly)
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 0.4
        #         color_closed = (0, 0, 255)
        #         thickness = 3
        #         cv2.rectangle(plot_img, (x + lx, y + ly), (x + lx + lw, y + ly + lh), color_closed, thickness)
        #         cv2.putText(plot_img, text_to_display, coordinates, font, font_scale, color_closed, thickness)

        # for (rx, ry, rw, rh) in right_eye:
        #     r_eye = face_img[ry:ry + rh, rx:rx + rw]
        #     # cv2.imwrite('right_eye.jpg', r_eye)
        #     r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        #     r_eye = cv2.resize(r_eye, (28, 28))
        #     r_eye = r_eye / 255
        #     r_eye = r_eye.reshape(28, 28, -1)
        #     r_eye = np.expand_dims(r_eye, axis=0)
        #     rpred = model_deep.predict_classes(r_eye)
        #     rpred_proba = rpred[0][0]
        #     if round(rpred_proba) == 1:
        #         text_to_display = f"Eye opening - {round(rpred_proba, 2)}"
        #         coordinates = (x + lx, y + ly)
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 1
        #         color = (255, 0, 0)
        #         thickness = 2
        #         cv2.putText(plot_img, text_to_display, coordinates, font, font_scale, color, thickness)
        #         break
        #     if round(rpred_proba) == 0:
        #         text_to_display = f"Eye opening - {round(rpred_proba, 2)}"
        #         coordinates = (x + lx, y + ly)
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 1
        #         color = (0, 0, 255)
        #         thickness = 2
        #         cv2.putText(plot_img, text_to_display, coordinates, font, font_scale, color, thickness)
        #         break

        #     cv2.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #
        # ret, frame_buff = cv2.imencode('.jpg', plot_img)
        # image_base64_cv = base64.b64encode(frame_buff)
        # image_ascii_cv = image_base64_cv.decode('ascii')
        # processed_image = f"data:image/jpeg;base64,{image_ascii_cv}"

    else:
        raw_image = None
        processed_image = None

    return render(request, 'webcam/image.html',
                  {'raw_image': raw_image, 'processed_image': processed_image})


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
