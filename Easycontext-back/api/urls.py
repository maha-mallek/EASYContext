from django.urls import include, path 
from django.conf.urls import url, include
from rest_framework import routers
from rest_framework.authtoken.views import obtain_auth_token  # <-- Here
from . import views

router = routers.DefaultRouter()
#router.register(r'users', views.UserViewSet)#register with django and view of all users

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    #path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),# ntestiw biha login django
    #path('api-token-auth/', obtain_auth_token, name='api_token_auth'),#not working but khaliha

    url(r'^auth/', include('rest_auth.urls')),#working for token
    path('users/register/', views.Registerpage, name="register"),#working 
    path('users/login/', views.api_login, name="login"),#working not used maghyr token hethy
    path('users/logout/', views.LogoutUser, name="logout"),
    url(r'^users/update/(?P<pk>[0-9]+)$',views.update_detail),


]