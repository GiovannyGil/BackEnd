from django.shortcuts import render, redirect
from .models import Curso
from django.contrib import messages
# Create your views here.


def home(request):
    cursos = Curso.objects.all() # me trae todos los registros del modelo cursos/tabla cursos
    messages.success(request, '¡Cursos Listados!') # genera una alerta con el mensaje
    return render(request, "gestionCursos.html", {"cursos" : cursos}) # me devuelve/retorna a la vista con los registros


def registrarCurso(request):
    codigo=request.POST['txtCodigo']
    nombre=request.POST['txtNombre']
    creditos=request.POST['numCredito']
    
    curso = Curso.objects.create(codigo=codigo, nombre=nombre, creditos=creditos)
    messages.success(request, '¡Cursos Registrado!')
    return redirect('/')

def edicionCurso(request, codigo):
    curso = Curso.objects.get(codigo=codigo)
    return render(request, 'edicionCurso.html', {"curso":curso})

def editarCurso(request):
    codigo=request.POST['txtCodigo']
    nombre=request.POST['txtNombre']
    creditos=request.POST['numCredito']
    
    curso = Curso.objects.get(codigo=codigo)
    curso.nombre = nombre
    curso.creditos = creditos
    curso.save()
    messages.success(request, '¡Cursos Actualizado!')
    return redirect('/')

def eliminarCurso(request, codigo):
    curso = Curso.objects.get(codigo=codigo)
    curso.delete()
    messages.success(request, '¡Cursos Eliminado!')
    return redirect('/')