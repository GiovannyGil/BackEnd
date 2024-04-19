from django.urls import path
from  ProyectoWebApp import views
from django.conf import settings
from django.conf.urls.static import static

# urls de las vistas de la app
urlpatterns = [
    path("",views.home ,name="Home"),
    
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) # vinvula para la ruta de los archivos multimedia en el admin 
