from .carro import Carro

def importe_total_carro(request):
    carro = Carro(request) # en caso de que deje de funcionar por "carro" descomenta esta linea
    total=0
    if request.user.is_authenticated: # si el usuario esta autenticado
        for key, value in request.session["carro"].items(): # para cada producto en el carro
            # total=total+float(value["precio"]) # incrementar el precio
            total=total+float(value["precio"])
    else:
        total = "Debes loguearte"
    return {"importe_total_carro":total}    


# def importe_total_carro(request):
#     total=0
#     #if request.user.is_authenticated:
#     for key, value in request.session["carro"].items():
#         total=total+float(value["precio"])
        
#     return {"importe_total_carro":total}