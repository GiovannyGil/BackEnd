from django.contrib import admin
from .models import Libro
# Register your models here.



admin.site.register(Libro) # se registra el model/tabla 'libro' en el panel de administrador

# crear usuarios
# py manage.py createsuperuser 
# ... "datos" ...
