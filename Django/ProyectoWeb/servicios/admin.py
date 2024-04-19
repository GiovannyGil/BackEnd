from django.contrib import admin

# Register your models here.


# importar los modelos del mismo directorio
from .models import Servicio

class ServicioAdmin(admin.ModelAdmin):
    readonly_fields=('created','updated')

admin.site.register(Servicio, ServicioAdmin)