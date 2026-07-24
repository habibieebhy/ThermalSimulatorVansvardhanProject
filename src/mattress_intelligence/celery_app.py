"""Celery application for distributed acquisition and analysis jobs."""

from __future__ import annotations

from celery import Celery

from .settings import Settings

settings = Settings()
# Redis redelivers unacknowledged tasks after the visibility timeout. Keep the
# timeout comfortably above the hard task limit even when an old .env provides
# an unsafe value.
visibility_timeout = max(
    settings.celery_visibility_timeout_seconds,
    settings.celery_task_time_limit_seconds + 3_600,
)

celery_app = Celery(
    "mattress_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["mattress_intelligence.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.celery_task_time_limit_seconds,
    task_soft_time_limit=max(60, settings.celery_task_time_limit_seconds - 60),
    task_always_eager=settings.celery_always_eager,
    task_store_eager_result=True,
    result_expires=settings.celery_result_expires_seconds,
    task_send_sent_event=True,
    worker_send_task_events=True,
    broker_transport_options={"visibility_timeout": visibility_timeout},
    result_backend_transport_options={"visibility_timeout": visibility_timeout},
    visibility_timeout=visibility_timeout,
)
