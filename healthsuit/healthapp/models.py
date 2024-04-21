from django.db import models

# Create your models here.
class Users(models.Model):
    Name=models.CharField(max_length=30)
    EmailID=models.CharField(max_length=30)
    Password=models.CharField(max_length=30)
    Address=models.TextField()
    PhoneNo=models.BigIntegerField()
    Place=models.CharField(max_length=30)
class doctors(models.Model):
    Name=models.CharField(max_length=30)
    Department=models.CharField(max_length=30)
    Pic=models.ImageField(upload_to='static/img/doctors/')
    Qualification=models.CharField(max_length=30)
    PhoneNo=models.BigIntegerField()
class contact(models.Model):
    Name=models.CharField(max_length=30)
    Email=models.CharField(max_length=30)
    Message=models.TextField()
