from django.urls import path
from  . import views

# urls de las vistas de la app
urlpatterns = [
    path("", views.procesar_pedido, name="Pedido"),
    
]