from django.shortcuts import render
from blog.models import Post, Categoria # importar el modelo servicios desde la app Servicio
# Create your views here

def blog(request):
    
    posts = Post.objects.all() # traiga todos los servicios  
    
    return render(request, "blog/blog.html", {"posts" : posts})


def categoria(request, categoria_id):
    categoria = Categoria.objects.get(id=categoria_id) # trae los posts con el id de la categoria seleccionado
    posts = Post.objects.filter(categorias=categoria) # filtra la categoria por categoria
    return render(request, "blog/categoria.html", {"categoria":categoria, "posts" : posts})
