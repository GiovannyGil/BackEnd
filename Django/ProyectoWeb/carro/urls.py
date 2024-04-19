from django.urls import path
from . import views

app_name="carro" # para llamar y usar las urls de forma más fácil

urlpatterns=[
    path("agregar/<int:producto_id>/", views.agregar_producto, name="Agregar"),
    path("eliminar/<int:producto_id>/", views.eliminar_producto, name="Eliminar"),
    path("restar/<int:producto_id>/", views.restar_producto, name="Restar"),
    path("limpiar/", views.limpiar_carro, name="Limpiar"),
]