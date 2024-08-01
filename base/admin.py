from django.contrib import admin
from django_celery_beat.admin import PeriodicTaskAdmin
from django_celery_beat.models import PeriodicTask
from base.tasks import run_data_script, run_retrain_script

# Unregister the existing admin
admin.site.unregister(PeriodicTask)

class CustomPeriodicTaskAdmin(PeriodicTaskAdmin):
    actions = PeriodicTaskAdmin.actions + ('run_task_now',)

    def run_task_now(self, request, queryset):
        for task in queryset:
            if task.name == 'run_data_script':
                run_data_script.apply_async()
            elif task.name == 'run_retrain_script':
                run_retrain_script.apply_async()
        self.message_user(request, "Tasks are being executed.")

# Register the model with the custom admin
admin.site.register(PeriodicTask, CustomPeriodicTaskAdmin)
