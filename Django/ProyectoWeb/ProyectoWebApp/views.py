from django.shortcuts import render

from carro.carro import Carro

# Create your views here.

# vistas/controlers
def home(request):
    carro = Carro(request)
    return render(request, "ProyectoWebApp/home.html")
