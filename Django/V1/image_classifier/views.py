from django.shortcuts import render
from .classify import predict, convertCategoryToName

# Create your views here.

def classify_image(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image']
        # Guardar la imagen subida temporalmente o procesarla directamente desde la memoria
        # Llamar a la función de clasificación
        probs, classes = predict(uploaded_image.path, model_from_file, topk=5)
        class_names = convertCategoryToName(classes)

        return render(request, 'result.html', {'probs': probs, 'class_names': class_names})

    return render(request, 'upload.html')