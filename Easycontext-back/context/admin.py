from django.contrib import admin

# Register your models here.
from .models import Contexte,Keywords
admin.site.register(Contexte)
admin.site.register(Keywords)
