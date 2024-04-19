from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from carro.carro import Carro
from pedidos.models import LineaPedido, Pedido
from django.contrib import messages
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
# Create your views here.

@login_required(login_url="/autenticacion/logear")
def procesar_pedido(request):
    pedido=Pedido.objects.create(user=request.user)
    carro=Carro(request)
    # guardar uno o mucbos productos en el pedido
    lineas_pedido=list() # guardar en una lista
    for key, value in carro.carro.items(): #recorrer elemento por elementos que se encuentre en el carro
        lineas_pedido.append(LineaPedido( # guardar la siguiente informacion por cada producto en el pedido del carro
            producto_id = key,  # clave
            cantidad = value["cantidad"], #cantidad
            user=request.user, # usuario logeado
            pedido=pedido # el pedido
        )) # agregar al final de la tabla o lista
        
    #guardar en la base de datos
    LineaPedido.objects.bulk_create(lineas_pedido) # guardar un lote, una lista o muchos registros
    enviar_mail(
        pedido=pedido,
        lineas_pedido=lineas_pedido,
        nombre_user=request.user.username,
        email_user=request.user.email,
    )
    messages.success(request, "El Pedido se Creo Correctamente")
    return redirect('../tienda')


def enviar_mail(**kwargs): # recibira un numero indeterminado de parametros
    asunto="Gracias Por Su Compra"
    mensaje=render_to_string("emails/pedido.html", {
        "pedido":kwargs.get("pedido"),
        "lineas_pedido":kwargs.get("lineas_pedido"),
        "nombre_user":kwargs.get("nombre_user"),
    })
    mensaje_texto = strip_tags(mensaje) # obmitir etiquetas html con strp_tags
    from_email="arvey950@gmail.com" # quien
    # to=kwargs.get("email_user") # para quien
    to="giogil2001@hotmail.com"
    # enviar el mail
    send_mail(asunto,mensaje_texto, from_email, [to], html_message=mensaje)