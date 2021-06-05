from django.urls import path, include

from webcam import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('image', views.image, name='image'),
    path('video', views.video, name='video'),
    ]