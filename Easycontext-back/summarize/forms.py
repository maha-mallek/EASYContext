
from django.forms import ModelForm
from django import forms
from .models import Summarize

class CreateSummaryForm(ModelForm):
    class Meta:
        model= Summarize
        fields = ['text','summarize']

