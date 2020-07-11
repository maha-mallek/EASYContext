from django.shortcuts import render
import os
# Create your views here.


import requests
from django.shortcuts import render
import requests
import sys
from subprocess import run,PIPE
from django.http import HttpResponse
from django.utils.translation import gettext

from .models import Summarize
from .serializers import SummarizeSerializer

from django.http import HttpResponse
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser 
from rest_framework import status
from .forms import  CreateSummaryForm

def button(request):

    return render(request,'summarize/home.html')

def output(request):
    data=requests.get("https://reqres.in/api/users/")
    #print(data.text)
    data=data.text
    return render(request,'summarize/home.html',{'data':data})

@csrf_exempt
def external(request):
    Summarize.objects.all().delete()
    inp= request.POST.get('param')
    direct=os.getcwd()
    out= run([sys.executable,direct+'//summarize.py',inp],shell=False,stdout=PIPE)

    ###
    try: 
        summarizers = Summarize.objects.get(pk=1) 
    except Summarize.DoesNotExist: 
        return HttpResponse(status=status.HTTP_404_NOT_FOUND) 
    pk=1
    summarizers = Summarize.objects.filter(pk=pk)   

    summarize_serializer = SummarizeSerializer(summarizers, many=True)
    
    return JsonResponse(summarize_serializer.data, safe=False)

    #return render(request,'summarize/home.html',{'data1':str(out.stdout)})

@csrf_exempt
def text(request):
  
  if request.method == 'POST':
      Summarize.objects.all().delete()

      summary_data = JSONParser().parse(request)
      
      inp=summary_data['text']
      
      direct=os.getcwd()
      out= run([sys.executable,direct+'//summarize.py',inp],shell=False,stdout=PIPE)

      try: 
        summarizers = Summarize.objects.get(pk=1) 
      except Summarize.DoesNotExist: 
        return HttpResponse(status=status.HTTP_404_NOT_FOUND) 
      pk=1
      summarizers = Summarize.objects.filter(pk=pk)   

      summarize_serializer = SummarizeSerializer(summarizers, many=True)
      
      return JsonResponse(summarize_serializer.data, safe=False)

     