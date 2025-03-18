from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
from .detector import detect_objects
import os

# Create your views here.
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        # Debugowanie
        file_path = fs.path(filename)
        print(f"File path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        processed_image_path, result = detect_objects(file_path)
        processed_image_url = settings.MEDIA_URL + 'uploads/' + os.path.basename(processed_image_path)

        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'processed_image_url': processed_image_url,
            'result': result
        })
    return render(request, 'upload.html')