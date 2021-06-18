from django.urls import path
from . import ws_consummers

websocket_urlpatterns = [
    path('ws/image', ws_consummers.WebcamConsumer.as_asgi()),
]
