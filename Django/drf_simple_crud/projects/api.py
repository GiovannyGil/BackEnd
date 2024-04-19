from .models import projects
from rest_framework import viewsets, permissions
from .serializers import ProjectSerializer

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = projects.objects.all() # traer todos los datos del modelo
    permission_classes = [permissions.AllowAny] # cualquier aplicacion cliente
    serializer_class = ProjectSerializer # coin esto API CREADA