class Carro:
    def __init__(self, request):
        self.request=request
        self.session=request.session # verificar la session
        carro=self.session.get("carro")# crea el carro si añade un producto
        if not carro:
            carro=self.session["carro"]={} # crear el carro vacio como una biblioteca vacia
        #else: # si ya hay carro
        self.carro=carro # si se va y regresa a la pagina, el carro permanezca igual si tenia productos
      
    def agregar(self, producto): # agregar productos al caroo
        if(str(producto.id) not in self.carro.keys()): # si el producto no está en el carro
            self.carro[producto.id]={ # agregarlo por primera vez
                "producto_id":producto.id,
                "nombre":producto.nombre,
                "precio": str(producto.precio), # pasar el precio a cadena o string
                "cantidad":1, # agregar de uno en uno al elegirlo si no elige cantidad
                "imagen":producto.imagen.url
            }
        else: # si ya está en el carro 
            for key, value in self.carro.items(): # por cada clave en el carro
                if key==str(producto.id): # comprobar si es igual a uno de los productos ya agregados
                    value["cantidad"]=value["cantidad"]+1 # si ya está, y lo vuelve a elegir, incrementar en 1
                    value["precio"]=float(value["precio"])+producto.precio
                    break
        self.guardar_carro() # guardar

    def guardar_carro(self):
        self.session["carro"]=self.carro # carro igual a la session del carro que se maneja
        self.session.modified=True # se modifico, entonces true
        
    def eliminar(self, producto):
        producto.id=str(producto.id) # pasar el id a string
        if producto.id in self.carro: # si el producto está en el carro
            del self.carro[producto.id] # eliminarlo del carro
            self.guardar_carro()
 
    def restar_producto(self, producto):
        for key, value in self.carro.items(): # por cada clave en el carro
                if key==str(producto.id): # comprobar si es igual a uno de los productos ya agregados
                    value["cantidad"]=value["cantidad"]-1 # si ya está, y lo vuelve a elegir, incrementar en 1
                    value["precio"]=float(value["precio"])-producto.precio
                    if value["cantidad"]<1: # si el producto es menor a 1
                        self.eliminar(producto) # eliminar el producto
                    break
        self.guardar_carro() # guardar    
        
    def limpiar_carro(self):
        self.session["carro"]={} # deja el carro limpio, en un diccionario vacio
        self.session.modified=True # comprueba la modificacion


        
