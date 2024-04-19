from django.db import models

# Create your models here.
class Persona(models.Model):
    id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=100)
    apellido = models.CharField(max_length=200)
    
    def __str__(self):
        return '{0},{1}'.format(self.apellido,self.nombre) # formatee y muestreme primero el apellido y luego nombre
    