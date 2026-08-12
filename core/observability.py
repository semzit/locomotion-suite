"""Common structured logging and tracing handles for application entry points."""

import structlog
from opentelemetry import trace

logger = structlog.get_logger("locomotion_suite")
tracer = trace.get_tracer("locomotion_suite")
