from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    # path('analysis/<int:analysis_id>/', views.get_analysis, name='get_analysis'),
]
