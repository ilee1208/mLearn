from django.urls import path, include
from django.contrib import admin
from stock import views

app_name = 'stock'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('stockhome/', views.stockhome, name='stockhome'),
    path('stockmodel/', views.stockmodel, name='stockmodel'),
    path('stocktrade/', views.stocktrade, name='stocktrade'),

]