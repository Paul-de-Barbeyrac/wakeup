import base64
from PIL import Image
from base64 import b64encode
import cv2
import numpy
import numpy as np
import io
from PIL import Image
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from webcam.camera import VideoCamera

face_cascade = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')


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

        cvimg = readb64(image_ascii)
        gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            # the higher the faster the detection but can be missing detection. Approx range 1.05 - 1.4
            minNeighbors=6,  # how many neighbors each candidate rectangle should have to retain it. Approx range 3 - 6
            minSize=(200, 200)  # objects smaller than that dimensions in pixels are ignored. Range depends on objects
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(cvimg, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.putText(cvimg, f"TOTO EST LA", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        ret, frame_buff = cv2.imencode('.jpg', cvimg)
        image_base64_cv = base64.b64encode(frame_buff)
        image_ascii_cv = image_base64_cv.decode('ascii')
        raw_image_cv = f"data:image/jpeg;base64,{image_ascii_cv}"
        processed_image = raw_image_cv

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
