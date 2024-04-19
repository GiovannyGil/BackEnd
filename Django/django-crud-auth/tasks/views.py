from django.shortcuts import render, redirect, get_object_or_404
# importar formalario de django para crear usuarios
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User  # registrar usuarios
from django.http import HttpResponse
from django.contrib.auth import login, logout, authenticate #cookie para el inicio de session
from .forms import TaskForm # importar el formulario creado para create task
from .models import Task # importar las tareas
from django.utils import timezone
from django.contrib.auth.decorators import login_required


# Create your views here.


def home(request):
    return render(request, 'home.html')


def signup(request):
    if request.method == 'GET':
        return render(request, 'signup.html', {
            'form': UserCreationForm,
        })
    else:
        if request.POST['password1'] == request.POST['password2']:
            try:
                # registrar usuario
                user = User.objects.create_user(username=request.POST['username'],
                                                password=request.POST['password1'])  # guarda los datos del formulario
                user.save() # guarda el registro
                login(request, user) # cookie para el inicio de session/validacion de seguridad
                return redirect('tasks') # redirige si guarda el nuevo usuario
            except:
                return render(request, 'signup.html', {
                    'form': UserCreationForm,
                    'error': 'Username alredy exist'
                })
        return render(request, 'signup.html', {
            'form': UserCreationForm,
            'error': 'Password do not match'
        })

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def tasks(request): # vista tareas
    tasks = Task.objects.filter(user=request.user, datecompleted__isnull=True) # muetsra las tareas del usuario logeado y las tareas aun no completadas
    return render(request, 'tasks.html', {'tasks':tasks})

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def tasks_completed(request): # vista tareas
    tasks = Task.objects.filter(user=request.user, datecompleted__isnull=False) # muetsra las tareas del usuario logeado y las tareas completadas
    return render(request, 'tasks.html', {'tasks':tasks})

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def task_detail(request, task_id):
    if request.method=='GET':
        task = get_object_or_404(Task, pk=task_id, user=request.user) # busque tareas por id y usuario logeado
        form = TaskForm(instance=task)
        return render(request, 'task_detail.html', {'task':task, 'form':form})
    else: 
        try:
            task = get_object_or_404(Task, pk=task_id, user=request.user) # busque tareas por id y usuario logeado
            form = TaskForm(request.POST, instance=task) # lo agrega a un formulario
            form.save() # actualiza
            return redirect('tasks')
        except ValueError:
            return render(request, 'task_detail.html', {'task':task, 'form':form, 'error':'error updating task'})

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def complete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.datecompleted = timezone.now()
        task.save()
        return redirect('tasks')

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def delete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.delete() # elimina la tarea
        return redirect('tasks')

@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def create_task(request):
    if request.method == 'GET':
        return render(request, 'create_task.html', {
            'form': TaskForm,
        })
    else:
        try:
            form = TaskForm(request.POST) # envia los datoos
            new_task = form.save(commit=False) 
            new_task.user = request.user # trae el usuario que asigno la tarea
            new_task.save() # guarda los datos
            return redirect('tasks')
        except ValueError:
            return render(request, 'create_task.html', {
                'form': TaskForm,
                'error':'Please provide valida data'
            })
   
@login_required # este elemento exige que se haya iniciado secion para aceder a la vista
def signout(request): # cerrar sessión
    logout(request)
    return redirect('home')


def signin(request):
    if request.method == 'GET':
        return render(request, 'signin.html', {
            'form': AuthenticationForm,
        })
    else: # traer los datos del formulario
        user = authenticate(request, username=request.POST['username'],
                     password=request.POST['password'])
        if user is None: # verifica si el username está vacio
            return render(request, 'signin.html', {
                'form': AuthenticationForm,
                'error': 'Username and/or password is incorrect',
            })
        else:
            login(request, user)
            return redirect('tasks')

