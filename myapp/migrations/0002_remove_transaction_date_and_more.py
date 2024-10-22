# Generated by Django 5.1.1 on 2024-10-17 09:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='transaction',
            name='date',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='isFlaggedFraud',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='newbalanceDest',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='newbalanceOrig',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='oldbalanceDest',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='oldbalanceOrg',
        ),
        migrations.AddField(
            model_name='transaction',
            name='time',
            field=models.TimeField(auto_now=True),
        ),
    ]
