from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Libro
from .forms import LibroForm
# Create your views here.

def inicio(request):
    return render(request, 'paginas/inicio.html')


def nosotros(request):
    return render(request, 'paginas/nosotros.html')

def libros(request):
    libros = Libro.objects.all() # trae la informacion de la tabla libros de la base de datos
    return render(request, 'libros/index.html', {'libros': libros} ) # envia los datos como parametros para usarlos

def crear(request):
    formulario = LibroForm(request.POST or None, request.FILES or None) # obtiene los datos del modelo para generar el formulario
    if formulario.is_valid(): # comprueba si el formulario es valido o correcto
        formulario.save() # lo guarda
        return redirect('libros') # redirije a la vista libros
    return render(request, 'libros/crear.html', {'formulario':formulario} ) # envia la informacion y creacion del formulario

def editar(request):
    return render(request, 'libros/editar.html')