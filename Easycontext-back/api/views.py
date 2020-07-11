from django.contrib.auth.models import User
from rest_framework import viewsets
from rest_framework import permissions
from .serializers import ClientSerializer , UserSerializer
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser ,FormParser, MultiPartParser
from rest_framework import status
from .models import *
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from .forms import CreateUserForm


from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import viewsets
from rest_framework import status

from rest_framework.views import APIView
from rest_framework.renderers import JSONRenderer
#this is register#######################################################"
@csrf_exempt
def Registerpage(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        user_data = JSONParser().parse(request)
        user_serializer = ClientSerializer(data=user_data)
        if user_serializer.is_valid():
            user_serializer.save()#userserilailer
            return JsonResponse(user_serializer.data, status=status.HTTP_201_CREATED)
        context={'form':form}
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#this is login########################################"
@api_view(['POST'])
@permission_classes([AllowAny])
def api_login(request):
    username = request.data['username']
    password = request.data['password']
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return Response(status=status.HTTP_200_OK)
    return Response(status=status.HTTP_400_BAD_REQUEST)

### for update inoformations
@csrf_exempt 
def update_detail(request, pk):
    try: 
        user = User.objects.get(pk=pk)
        
    except User.DoesNotExist: 
        return HttpResponse(status=status.HTTP_404_NOT_FOUND) 
 
    if request.method == 'PUT': 
        user_data = JSONParser().parse(request)
        
        user_serializer = UserSerializer(user, data=user_data) 
        if user_serializer.is_valid(): 
            user_serializer.save() 
            
            return JsonResponse(user_serializer.data) 
        return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
 
   













#work for the register with django
class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """ 
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = ClientSerializer
    
    #serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]



#working , khalihem hethom login w register w loguot django khaw
def LogoutUser(request):
    logout(request)
    return redirect ('login')


def Registerpagee(request):
   #if request.user.is_authenticated:
    #    return redirect('home')
    #else:
        form = CreateUserForm()

        if request.method == "POST":
            form = CreateUserForm(request.POST)#UserCreationForm howa formulaire django par defaut
            if form.is_valid():
                form.save()
                return redirect('login')

        context={'form':form}
        return render(request, 'accounts/register.html',context)

@csrf_exempt
def Loginpage(request):
    if request.method == "POST":
        username= request.POST.get('username')
        password = request.POST.get('password')

     

        user= authenticate(request, username=username, password=password)

        if user is not None:
            login(request,user)
            return JsonResponse(user_serializer.data, status=status.HTTP_201_CREATED)
            
       

    context={}
    return JsonResponse(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def Loginpagee(request):
   # if request.user.is_authenticated:
    #    return redirect('home')
   # else :

        if request.method == "POST":
            username= request.POST.get('username')
            password = request.POST.get('password')

            user= authenticate(request, username=username, password=password)

            if user is not None:
                login(request,user)
                return redirect ('home')


        context={}

        return render(request, 'accounts/login.html',context)