# serializar es convertir data a un archivo para intercambiar informacion (api) entre proyectos


from rest_framework import serializers
from .models import Persona


class PersonaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Persona 
        fields=[
            'id',
            'nombre',
            'apellido'
        ]