from django.shortcuts import render, redirect
from .forms import FormularioContacto # importar el formulario diseñado en el archivo forms
from django.core.mail import EmailMessage # importar clase para enviar correos

# Create your views here.



def contacto(request):
    Formulario_Contacto = FormularioContacto() # instanciar el formulario
    
    if request.method == "POST": # si se envia informacion mediante POST
        Formulario_Contacto=FormularioContacto(data=request.POST)    # guarda la informacion o eniala al formulario
        if Formulario_Contacto.is_valid(): # si el formulario es valido
            nombre = request.POST.get("nombre") # almacena lo enviado en el campo ... en la variable ...
            email = request.POST.get("email")
            contenido = request.POST.get("contenido")
            
            
            # enviar correo
            email=EmailMessage(
                "Mesnaje de App Django", # mensaje de envio con los datps
                "El usuarios {} con la direccion {} escribe lo siguiente: \n\n {}".format(nombre,email,contenido),
                "", ["Arvey950@gmail.com"],reply_to=[email]) # de donde viene y si desea responder
            
            try: # intenta enviar
                email.send() # si esta bien, se envia
            
                return redirect("/contacto/?valido")
            except: # si sale un error o no se envia
                return redirect("/contacto/?Novalido")
    
    return render(request, "contacto/contacto.html", {"miFormulario":Formulario_Contacto})