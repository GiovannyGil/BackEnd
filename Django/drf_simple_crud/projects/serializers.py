from rest_framework import serializers
from .models import projects


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = projects # Conectar el modulo
        fields = ('id','title','description', 'technology', 'created_at') # campos que se puede usar en eñ API
        read_only_fields = ('created_at',) # campos de solo lectura