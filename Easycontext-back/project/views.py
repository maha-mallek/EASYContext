from django.shortcuts import render 
import requests

def output (request):
   

    data=request.get("https://reqres.in//api/users")
    print (data.text)
    return render(request,'home.html')