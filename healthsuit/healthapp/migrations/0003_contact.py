# Generated by Django 5.0.3 on 2024-03-20 10:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('healthapp', '0002_doctors'),
    ]

    operations = [
        migrations.CreateModel(
            name='contact',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Name', models.CharField(max_length=30)),
                ('Email', models.CharField(max_length=30)),
                ('Message', models.TextField()),
            ],
        ),
    ]
