from django.forms import ModelForm
from .models import Task # importar el modelo a usar para el formulario

# crear formulario para crear tareas "tasks"
class TaskForm(ModelForm):
    class Meta:
        model = Task # el formulario esta basado en el modelo Task
        fields = ['title', 'description', 'important']