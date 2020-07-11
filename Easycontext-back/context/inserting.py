from models import * 
import pandas as pd 
import django
import os 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()


"""
class document(models.Model):
    id_user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic = models.CharField(max_length=1000 , blank=False , default='')
    Lien = models.CharField(max_length=1000 , blank=False , default='' )

class Contexte(models.Model):
    etiquette = models.CharField(max_length=1000,blank=False,default='')

class keywords(models.Model):
    id_contexte = models.ForeignKey(Contexte , on_delete=models.CASCADE)
    mots = models.CharField(max_length=100 , blank=False , default='')
    score = models.IntegerField(default=0 , blank=False)
"""
from django.contrib.auth.models import User

user= User.objects.get(pk=1)
for context in etiquettes:
    print(context)
    contexte = Contexte(etiquettes=context)
    contexte.save()
import datetime
i=Contexte.object.get(pk=i)
i=1
for titl in title: 
    for text in textes:
        texte = document(id_user=user,topic_id = i  ,Texte = text ,  Date=datetime.datetime.now())
        texte.save()
        i=i+1






