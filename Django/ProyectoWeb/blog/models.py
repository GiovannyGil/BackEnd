from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class Categoria(models.Model):
    nombre =  models.CharField(max_length=50)
    created = models.DateTimeField(auto_now_add=True) # tome la fecha automaticamente
    updated  = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name="categoria"
        verbose_name_plural="categorias"

    def __str__(self):
        return self.nombre


class Post(models.Model):
    titulo =  models.CharField(max_length=50)
    contenido = models.CharField(max_length=50)
    imagen = models.ImageField(upload_to='blog', null=True, blank=True) # imagen opcional, puede quedar nulo o vacio
    autor=models.ForeignKey(User, on_delete=models.CASCADE) # usuarios que publica, y si el usuario es eliminado, las publicaciones tambien
    categorias = models.ManyToManyField(Categoria) # relacion de muchos a muchos con categorias 
    created = models.DateTimeField(auto_now_add=True) # tome la fecha automaticamente
    updated  = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name="post"
        verbose_name_plural="posts"

    def __str__(self):
        return self.titulo

