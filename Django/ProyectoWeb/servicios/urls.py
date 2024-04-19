from django.urls import path
from  . import views
from django.conf import settings
from django.conf.urls.static import static

# urls de las vistas de la app
urlpatterns = [
    path("", views.servicios, name="Servicios"),
]