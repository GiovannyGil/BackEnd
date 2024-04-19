from django.shortcuts import render, redirect
from django.views.generic import View
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm # formulario para crear un usuario

# Create your views here.


class VRegistro(View):
    def get(self, request):
        form=UserCreationForm() # traer el formulario
        return render(request, "registro/registro.html", {"form":form})
    
    def post(self, request):
        form=UserCreationForm(request.POST) # traer el formulario
        
        #comprobar si el registro salió bien
        if form.is_valid():
            usuario=form.save() # guardar
            login(request, usuario) # iniciar session cuando se registre
            return redirect("Home")
        else:
            # recorrer los errores posibles
            for msg in form.error_messages:
                messages.error(request, form.error_messages[msg]) # mostrar los errores
            return render(request, "registro/registro.html", {"form":form})


def cerrar_sesion(request):
    logout(request)
    return redirect("Home")


def logear(request):
    #validar datos
    if request.method=="POST":
        form=AuthenticationForm(request, data=request.POST) # guardar la informacion
        if form.is_valid(): #verificar su es valida
            nombre_usuario=form.cleaned_data.get("username") # traer la informacion del input --> usarname
            contra=form.cleaned_data.get("password")
            usuario=authenticate(username=nombre_usuario, password=contra) # verificar que los datos sean iguales
            if usuario is not None: # si es correcto
                login(request, usuario) # iniciar session
                return redirect("Home") # ir a home
            else:
                messages.error(request, "usuario no valido")
        else:
            messages.error(request, "información incorrecta")
    form = AuthenticationForm()# llamar la vista de login
    return render(request, "login/login.html", {"form":form})
    
    
    
    
# def autenticacion(request):
    
#     return render(request, "registro/registro.html")

