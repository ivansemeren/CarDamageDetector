# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2020-01-26 00:58
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PicUpload',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('imagefile', models.ImageField(blank=True, upload_to='pic_upload')),
            ],
        ),
    ]
