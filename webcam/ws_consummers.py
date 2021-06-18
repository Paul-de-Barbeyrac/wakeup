import base64
import io
from io import StringIO
import numpy as np
from PIL import Image
import cv2
import imutils
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.serializers.json import DjangoJSONEncoder
from django.core.exceptions import ObjectDoesNotExist
import json




class WebcamConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        """
        Websocket Connection
        check if the client try to connect to a valid site by checking if site exist into the database
        other case we reject the connection
        """
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        """
        catch data send by a client
        :param text_data: string type of request
        :param bytes_data: not used
        :return: json requested data
        """
        print(text_data)
        print(bytes_data)
        if text_data:
            sbuf = StringIO()
            sbuf.write(text_data)

            # decode and convert into image
            b = io.BytesIO(base64.b64decode(text_data))
            pimg = Image.open(b)

            ## converting RGB to BGR, as opencv standards
            frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

            # Process the image frame
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            imgencode = cv2.imencode('.jpg', frame)[1]

            # base64 encode
            stringData = base64.b64encode(imgencode).decode('ascii')
            b64_src = 'data:image/jpg;base64,'
            stringData = b64_src + stringData

            # emit the frame back
            await self.send(text_data=stringData)

        else:
            await self.send(text_data='toto')

    async def disconnect(self, close_code):
        pass
