from django.contrib.auth.models import User
from rest_framework import serializers


#this is the serializer
class ClientSerializer(serializers.ModelSerializer):

    class Meta:
        model = User
        fields =  '__all__'
        #fields = ('id','username','email','password')
        extra_kwargs = {'password' : {'write_only':True ,'required':True}}

    def create(self , validated_data):
        user = User.objects.create_user(**validated_data)
        return user






# used for update

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['username','email']
        

