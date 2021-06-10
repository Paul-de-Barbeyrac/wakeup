import base64
import io

import cv2
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
    processed_image = None
    image_opencv = None
    if request.method == 'POST':
        image = request.FILES['file'].read()
        image_base64 = base64.b64encode(image)
        image_ascii = image_base64.decode('ascii')
        raw_image = f"data:image/jpeg;base64,{image_ascii}"

        lpred_output = "Unknown"
        rpred_output = "Unknown"

        raw_img = readb64(image_ascii)
        raw_img_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        plot_img = raw_img

        faces = face_cascade.detectMultiScale(
            raw_img_gray,
            scaleFactor=1.3,
            # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
            minNeighbors=3,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
            minSize=(30, 30)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
        )

        for (x, y, w, h) in faces:
            cv2.imwrite('raw_image.jpg', raw_img)
            cv2.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

            face_img = raw_img[y:y + h, x:x + w]
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(str(x) + str(y) + str(w) + str(h) + '_faces.jpg', raw_img)

            left_eye = left_eye_cascade.detectMultiScale(
                face_img_gray,
                scaleFactor=1.4,
                minNeighbors=5,
                minSize=(30, 30)
            )

            right_eye = right_eye_cascade.detectMultiScale(
                face_img_gray,
                scaleFactor=1.4,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (lx, ly, lw, lh) in left_eye:
                cv2.rectangle(plot_img, (x + lx, y + ly), (x + lx + lw, y + ly + lh), (0, 0, 255), 3)

                l_eye = face_img[ly:ly + lh, lx:lx + lw]
                cv2.imwrite('left_eye.jpg', l_eye)
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

            for (rx, ry, rw, rh) in right_eye:
                cv2.rectangle(plot_img, (x + rx, y + ry), (x + rx + rw, y + ry + rh), (0, 255, 0), 3)

                r_eye = face_img[ry:ry + rh, rx:rx + rw]
                cv2.imwrite('right_eye.jpg', r_eye)
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

        cv2.putText(plot_img, f"Model predict LEFT eye : {lpred_output}", (0, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255),
                    1)
        cv2.putText(plot_img, f"Model predict RIGHT eye : {rpred_output}", (0, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0),
                    1)

        ret, frame_buff = cv2.imencode('.jpg', plot_img)
        image_base64_cv = base64.b64encode(frame_buff)
        image_ascii_cv = image_base64_cv.decode('ascii')
        processed_image = f"data:image/jpeg;base64,{image_ascii_cv}"

    else:
        raw_image = None

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
