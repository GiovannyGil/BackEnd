from django.shortcuts import render, redirect
from .carro import Carro # importar la calse carro
from tienda.models import Producto # importar el modelp/tabla de productos



# Create your views here.



def agregar_producto(request, producto_id):
    carro = Carro(request) # crear o instanciar la clase carro
    producto = Producto.objects.get(id=producto_id) # obtener el producto que se desea agregar
    carro.agregar(producto=producto)
    return redirect("Tienda")


def eliminar_producto(request, producto_id):
    carro = Carro(request) # crear o instanciar la clase carro
    producto = Producto.objects.get(id=producto_id) # obtener el producto que se desea agregar
    carro.eliminar(producto=producto)
    return redirect("Tienda")



def restar_producto(request, producto_id):
    carro = Carro(request) # crear o instanciar la clase carro
    producto = Producto.objects.get(id=producto_id) # obtener el producto que se desea agregar
    carro.restar_producto(producto=producto)
    return redirect("Tienda")


def limpiar_carro(request, producto_id):
    carro = Carro(request) # crear o instanciar la clase carro
    carro.limpiar_carro()
    return redirect("Tienda")

