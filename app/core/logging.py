"""Structured logging with source locations and credential-safe tracebacks."""

import os
import re
import sys
import traceback

from loguru import logger


def safe_error_text(value: object) -> str:
    text = str(value)
    for name, secret in os.environ.items():
        if len(secret) >= 8 and any(word in name.upper() for word in
                                   ('KEY', 'TOKEN', 'PASSWORD', 'SECRET', 'CONNECTION_STRING')):
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)(bearer\s+)[\w.\-+/=]+', r'\1[REDACTED]', text)
    return re.sub(
        r'''(?ix)((?:api[_-]?key|access[_-]?token|password|secret|authorization)"?\s*[:=]\s*["']?)([^\s,"'}]+)''',
        r'\1[REDACTED]', text,
    )


def log_error(error: Exception, *, service: str, **context) -> None:
    frames = traceback.extract_tb(error.__traceback__)
    origin = frames[-1] if frames else None
    source_frames = list(traceback.walk_tb(error.__traceback__))
    module = source_frames[-1][0].f_globals.get("__name__") if source_frames else None
    # format_tb contains source locations, not local-variable values or exception bodies.
    trace = ''.join(traceback.format_tb(error.__traceback__))
    logger.bind(
        service=service, error_type=type(error).__name__,
        error_module=module, error_file=origin.filename if origin else None,
        error_function=origin.name if origin else None,
        error_line=origin.lineno if origin else None,
        traceback=safe_error_text(trace), **context,
    ).error('{}: {}', type(error).__name__, safe_error_text(error))


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stdout, level=level, serialize=True, backtrace=False, diagnose=False,
               format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | '
                      '{name}:{function}:{line} | {message}')
