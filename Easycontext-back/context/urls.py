from django.urls import include, path 
from django.conf.urls import url, include
from rest_framework import routers
from . import views


urlpatterns = [
 url(r'^extra', views.external),
 url(r'^add',views.add),

 #added
 #url(r'^context',views.context),
 url(r'^lda',views.lda),
 url(r'^download/(?P<pk>[0-9]+)$',views.recent_result),
 url(r'^downloadkey/(?P<pk>[0-9]+)$',views.keywords_result),
]
