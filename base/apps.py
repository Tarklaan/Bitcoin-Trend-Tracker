from django.apps import AppConfig
from django.db.models.signals import post_migrate

class BaseConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "base"

    def ready(self):
        from django_celery_beat.models import PeriodicTask, CrontabSchedule

        def setup_periodic_tasks(sender, **kwargs):
            # Data script every day at 1:00 AM
            schedule_data, _ = CrontabSchedule.objects.get_or_create(
                minute='0', hour='1', day_of_week='*', day_of_month='*', month_of_year='*'
            )
            PeriodicTask.objects.get_or_create(
                crontab=schedule_data, name='Run data script every day', task='base.tasks.run_data_script'
            )

            # Retrain script every 3 days at 11:30 AM
            schedule_retrain, _ = CrontabSchedule.objects.get_or_create(
                minute='30', hour='11', day_of_week='*/3', day_of_month='*', month_of_year='*'
            )
            PeriodicTask.objects.get_or_create(
                crontab=schedule_retrain, name='Run retrain script every 3 days', task='base.tasks.run_retrain_script'
            )

        post_migrate.connect(setup_periodic_tasks, sender=self)
