import cv2


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_flip = cv2.flip(image, 1)  # flip vertically
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()
