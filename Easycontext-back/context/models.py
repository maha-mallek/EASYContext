from django.db import models
from django.contrib.auth.models import User
from datetime import date

# Create your models here.
class Contexte(models.Model):
    etiquette = models.CharField(max_length=1000,blank=False,default='')

    def __str__(self):
        return self.etiquette

class document(models.Model):
    id_user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic_id = models.ForeignKey(Contexte, on_delete=models.CASCADE)
    Text=models.TextField(max_length=15000,blank=False,default='')
    Date = models.DateTimeField(auto_now=True)


class Keywords(models.Model):
    id_etiquette = models.ForeignKey(Contexte , on_delete=models.CASCADE)
    mots = models.CharField(max_length=100 , blank=False , default='')

    def __str__(self):
        return self.mots
