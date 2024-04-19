from django import forms
from .models import Libro

# crear un formulario en base al modelo

class LibroForm(forms.ModelForm):
    class Meta():
        model = Libro
        fields = '__all__'