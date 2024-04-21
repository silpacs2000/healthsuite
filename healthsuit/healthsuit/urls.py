"""
URL configuration for healthsuit project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from healthapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('registration',views.user_registration,name='registration'),
    path('login',views.user_login,name='login'),
    path('logout',views.user_logout,name='logout'),
    path('heart',views.heart_prediction,name='heart'),
    path('myprfile',views.myprofile,name='myprofile'),
    path('liver',views.liver_prediction,name='liver'),
    path('lung',views.lung_prediction,name='lung'),
    path('departments',views.departments,name='departments'),
    path('doctors/<str:department>',views.doctorslist,name='doctors'),
    path('updateprofile/<int:id>',views.updateprofile,name='updateprofile'),
    path('contactus',views.contactus,name='contactus'),
]
