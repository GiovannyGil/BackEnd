from django.shortcuts import render
from servicios.models import Servicio # importar el modelo servicios desde la app Servicio
# Create your views here.


def servicios(request):
    
    
    servicios = Servicio.objects.all() # traiga todos los servicios  
    
    return render(request, "servicios/servicios.html", {"servicios":servicios})
