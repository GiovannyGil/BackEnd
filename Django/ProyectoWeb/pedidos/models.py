from django.db import models
from django.db.models import F, Sum, FloatField
from django.contrib.auth import get_user_model

from tienda.models import Producto
# Create your models here.

User=get_user_model() # llamar el usuario activo

class Pedido(models.Model):
    
    user= models.ForeignKey(User, on_delete=models.CASCADE) # llave foranea de usuario con usuario activo, y eliminacion en cascada
    created_at= models.DateTimeField(auto_now_add=True)
    
    
    def __str__(self):
        return self.id
    
    @property
    # propiedad, total del pedido
    def total(self):
        return self.lineapedido_set.aggregate(
            total=Sum(F("precio")*F("cantidad"), output_field=FloatField())
        )["total"] # traer el total de el pedido
    
    
    class Meta:
        db_table='pedidos'
        verbose_name='pedido'
        verbose_name_plural='pedidos'
        ordering=['id']
        
        
class LineaPedido(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE) # llave foranea de producto, y eliminacion en cascada
    producto = models.ForeignKey(Producto, on_delete=models.CASCADE)
    pedido = models.ForeignKey(Pedido, on_delete=models.CASCADE)
    cantidad = models.IntegerField(default=1)
    created_at= models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f'{self.cantidad} unidades de {self.producto.nombre}'
    
    class Meta:
        db_table='LineaPedidos'
        verbose_name='LineaPedido'
        verbose_name_plural='LineasPedidos'
        ordering=['id']