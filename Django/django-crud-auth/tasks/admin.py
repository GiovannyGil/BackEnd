from django.contrib import admin
from .models import Task # importar el modelo



class TaskAdmin(admin.ModelAdmin): # añadir campos de solo lectura al panel, 
    #los que son automaticos que no se pueden modificar
    readonly_fields = ('created', )

# Register your models here.
admin.site.register(Task, TaskAdmin) # registrar este modelo/tabla en el panel administrador para manejarlo allí