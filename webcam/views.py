import base64
from base64 import b64encode

import cv2
import numpy
import numpy as np
import io
from PIL import Image
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from webcam.camera import VideoCamera
from webcam.form import UploadFileForm


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
        form = UploadFileForm(request.FILES['file'])

        print(raw_image)

        base64_decoded = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.array(image)
        print(image_np.shape)

        image_opencv = Image.fromarray(image_np.astype(np.uint8))
        # Plante ici. A revoir
        image_opencv_base64 = base64.b64encode(image_opencv)
        image_opencv_ascii = image_opencv_base64.decode('ascii')
        processed_image = f"data:image/jpeg;base64,{image_opencv_ascii}"

        print(processed_image)

    else:
        form = UploadFileForm()
        raw_image = None

    return render(request, 'webcam/image.html',
                  {'form': form, 'raw_image': raw_image, 'processed_image': processed_image})


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
