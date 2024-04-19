from django.db import models

# Create your models here.
class Libro(models.Model):
    id = models.AutoField(primary_key=True)
    titulo = models.CharField(max_length=100, verbose_name="Titulo")
    imagen = models.ImageField(upload_to='imagenes/', verbose_name="Imagen", null=True)
    descripcion = models.TextField(verbose_name="Descripción", null=True)
    
    def __str__(self):
        fila = "Titulo: " + self.titulo + " - " + self.descripcion
        return fila
    
    def delete(self, using=None, keep_parents=False): # funcion para cuando se elimine un registro, la imagen se borre correctamente
        self.imagen.storage.delete(self.imagen.name)
        super().delete()    
        
    
    
# en consola 'py manage.py makemigrations'
# para generar la tabla/modelo en la base de datos con todas las migraciones adicionales
# luego para generar la migracion = 'py manage.py migrate'
