from django.shortcuts import render
from rest_framework import generics
from .models import Persona
from .seriealizer import PersonaSerializer

# para crear el login
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
# from django.decorators.csrf import csrf_protect
# from django.contrib.generic.edit import FormView
from django.contrib.auth import login,logout
from django.http import HttpResponseRedirect
from django.contrib.auth.forms import AuthenticationForm


# Create your views here.
class PersonaList(generics.ListCreateAPIView):
    queryset = Persona.objects.all()#trear todos los objetos de el modelo ya serializado de Persona
    serializer_class = PersonaSerializer
    
    
class Login():
    templete_name = "login.html"
    form_class = AuthenticationForm
    succes_url = reverse_lazy('persona:persona_list')
    
    # @method_decorator(csrf_protect)
    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return HttpResponseRedirect(self.get_success_url())
        else:
            return super(Login,self).dispatch(self,*args,*kwargs)

