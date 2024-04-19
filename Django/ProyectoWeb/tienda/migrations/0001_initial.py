# Generated by Django 4.2.2 on 2023-06-30 17:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CategoriaPro",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("nombre", models.CharField(max_length=50)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "categoriaPro",
                "verbose_name_plural": "categoriasPro",
            },
        ),
        migrations.CreateModel(
            name="Producto",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("nombre", models.CharField(max_length=50)),
                (
                    "imagen",
                    models.ImageField(blank=True, null=True, upload_to="tienda"),
                ),
                ("precio", models.FloatField()),
                ("disponibilidad", models.BooleanField(default=True)),
                (
                    "Categorias",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="tienda.categoriapro",
                    ),
                ),
            ],
            options={
                "verbose_name": "Producto",
                "verbose_name_plural": "Productos",
            },
        ),
    ]
