import base64
import io
import time

import cv2
import dlib
import numpy as np
from PIL import Image
from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from tensorflow import keras
from skimage import exposure
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

def socket(request):
    return render(request, 'webcam/socket.html')


def image(request):
    raw_image = None
    processed_image = None
    image_opencv = None
    if request.method == 'POST':

        image = request.FILES['file'].read()
        image_base64 = base64.b64encode(image)
        image_ascii = image_base64.decode('ascii')
        raw_image = f"data:image/jpeg;base64,{image_ascii}"

        raw_img = readb64(image_ascii)

        # Convert to dlib
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # dlib face detection
        detector = dlib.get_frontal_face_detector()
        detections = detector(img, 1)

        # Find landmarks
        start = time.time()
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        stop = time.time()
        # print(stop - start)
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

        for eye in left_eyes:
            width = max(eye, key=lambda x: x[0])[0] - min(eye, key=lambda x: x[0])[0]
            center_x = round((min(eye, key=lambda x: x[0])[0] + max(eye, key=lambda x: x[0])[0]) / 2)
            center_y = round((min(eye, key=lambda x: x[1])[1] + max(eye, key=lambda x: x[1])[1]) / 2)

            # Plot center of eye
            # cv2.circle(imgd, (center_x, center_y), 2, (0, 255, 0), -1)
            extract_eye_left = imgd[center_y - width:center_y + width, center_x - width:center_x + width]
            extract_eye_left = cv2.cvtColor(extract_eye_left, cv2.COLOR_BGR2GRAY)
            extract_eye_left = cv2.resize(extract_eye_left, (28, 28))

            # Ne marche pas mais bonne idee. Faudrait reentrainer le model
            # p2, p98 = np.percentile(extract_eye_left, (10, 90))
            # extract_eye_left = exposure.rescale_intensity(extract_eye_left, in_range=(p2, p98))

            # Save image
            cv2.imwrite(f"left2-eye-{center_x}-{center_y}.jpg", extract_eye_left)

            extract_eye_left = extract_eye_left / 255
            extract_eye_left = extract_eye_left.reshape(28, 28, -1)
            extract_eye_left = np.expand_dims(extract_eye_left, axis=0)
            pred = model_deep.predict(extract_eye_left)
            proba = pred[0][0]
            color = (0, 255 * proba, 255 * (1 - proba))
            # print(round(proba, 4))
            cv2.rectangle(imgd, (center_x - width, center_y - width), (center_x + width, center_y + width), color, 3)

            text_to_display = f"{round(float(proba), 2)}"
            coordinates = (center_x - width, center_y - width - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 1
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
            cv2.imwrite(f"right-eye-{center_x}-{center_y}.jpg", extract_eye_right)

            extract_eye_right = extract_eye_right / 255
            extract_eye_right = extract_eye_right.reshape(28, 28, -1)
            extract_eye_right = np.expand_dims(extract_eye_right, axis=0)
            pred = model_deep.predict(extract_eye_right)
            proba = pred[0][0]
            color = (0, 255 * proba, 255 * (1 - proba))
            # print(round(proba, 4))
            cv2.rectangle(imgd, (center_x - width, center_y - width), (center_x + width, center_y + width), color, 3)

            text_to_display = f"{round(float(proba), 2)}"
            coordinates = (center_x - width, center_y - width - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            cv2.putText(imgd, text_to_display, coordinates, font, font_scale, color, thickness)

        for i in bb:
            cv2.rectangle(imgd, i[0], i[1], (255, 0, 0), 5)  # Bounding box

        ret, frame_buff = cv2.imencode('.jpg', imgd)
        image_base64_cv = base64.b64encode(frame_buff)
        image_ascii_cv = image_base64_cv.decode('ascii')
        processed_image = f"data:image/jpeg;base64,{image_ascii_cv}"



    else:
        raw_image = None
        processed_image = None


    return render(request, 'webcam/image.html',
                  {'raw_image': raw_image, 'processed_image': processed_image})


def gen(camera):
    while True:
        # start=time.time()
        frame = camera.get_frame()
        # print(time.time()-start)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector= dlib.get_frontal_face_detector()
    return StreamingHttpResponse(gen(VideoCamera(sp,detector)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
