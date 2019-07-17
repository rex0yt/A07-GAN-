from django.urls import path
from . import views

urlpatterns = [
    path('show/', views.receive_num),
    path('shibie/',views.shibie),
    path('getData/', views.scene),
    path('code/', views.receive_code)
]