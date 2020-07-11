from rest_framework import serializers 
from .models import *
from summarize.serializers import * 
class KeywordSerializer(serializers.ModelSerializer):
        class Meta:
            model = Keywords
            fields = ['mots']

class ContexteSerializer(serializers.ModelSerializer):
        class Meta:
            model = Contexte
            fields = ['etiquette']

class DocumentSerializer(serializers.ModelSerializer):
        class Meta:
            model = document
            fields = ['id_user','topic_id','Text','Date']



