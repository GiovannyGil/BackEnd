# Generated by Django 4.2.3 on 2023-07-21 14:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('projects', '0002_projects_delete_prohects'),
    ]

    operations = [
        migrations.RenameField(
            model_name='projects',
            old_name='descriptio',
            new_name='description',
        ),
    ]
