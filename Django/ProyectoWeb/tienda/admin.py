from django.contrib import admin
from .models import CategoriaPro, Producto

# Register your models here.


class CategoriaProAdmin(admin.ModelAdmin):
    readonly_fields=("created", "updated")
    
class ProductoAdmin(admin.ModelAdmin):
    readonly_fields=("created", "updated")

admin.site.register(CategoriaPro, CategoriaProAdmin)

admin.site.register(Producto, ProductoAdmin)