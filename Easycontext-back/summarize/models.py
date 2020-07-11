from django.db import models

# Create your models here.

class Summarize(models.Model):
    id=models.IntegerField( blank=False,default=1,primary_key=True)
    text=models.TextField(max_length=20000 , blank=False , default='' )
    summarize=models.TextField(max_length=15000 , blank=False , default='' )
