from django.db import models
from django.contrib.auth.models import User



# Create your models here.
class Task(models.Model):
    title = models.CharField(max_length=100) # titulo de 100 caracteres maxim0
    description = models.TextField(blank=True) # descripcion que puede estar vacio
    created = models.DateTimeField(auto_now_add=True) # fecha que se agrega automaticamente
    datecompleted = models.DateTimeField(null=True, blank=True) # fecha que se agrega manualmente y es opcional
    important = models.BooleanField(default=False) # campo boleando por defecto falso, no todas las tareas son importantes
    user = models.ForeignKey(User, on_delete=models.CASCADE) # relacion con tabla usuarios y eliminacion en cascada si el usuario es eliminado
    
    ## como se mostrara en el panel administrador == campos en vista rápida
    def __str__(self):
        return self.title + '- by: ' + self.user.username