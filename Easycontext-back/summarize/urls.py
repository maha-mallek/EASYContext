from django.urls import include, path 
from django.conf.urls import url, include
from rest_framework import routers
from . import views

urlpatterns = [
   url(r'^button', views.button),
   url(r'^output', views.output,name="script"),
   url(r'^external', views.external,name="external"),#/(?P<pk>[0-9]+)/$
   url(r'^text', views.text),



]
