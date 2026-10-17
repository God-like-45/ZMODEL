"""Smoke and import tests for all UI views and components."""

import pytest
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.system_info import get_system_environment
from zmodel.ui.components import (
    format_bytes,
    render_metric_box,
    render_warning_item,
    render_workspace_header,
)
from zmodel.ui.theme import apply_theme
from zmodel.ui.views.benchmark import render_benchmark_view
from zmodel.ui.views.compatibility import render_compatibility_view
from zmodel.ui.views.history import render_history_view
from zmodel.ui.views.inspector import render_inspector_view
from zmodel.ui.views.optimization import render_optimization_view
from zmodel.ui.views.overview import render_overview_view
from zmodel.ui.views.settings import render_settings_view
from zmodel.ui.views.upcoming import render_upcoming_view
from zmodel.ui.views.validation import render_validation_view
from zmodel.ui.views.workers import render_workers_view


def test_ui_imports():
    assert callable(render_overview_view)
    assert callable(render_inspector_view)
    assert callable(render_compatibility_view)
    assert callable(render_benchmark_view)
    assert callable(render_optimization_view)
    assert callable(render_history_view)
    assert callable(render_settings_view)
    assert callable(render_workers_view)
    assert callable(render_validation_view)
    assert callable(render_upcoming_view)
    assert callable(render_warning_item)
    assert callable(render_workspace_header)
    assert callable(format_bytes)
    assert callable(apply_theme)


def test_format_bytes_units():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.00 MB"
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.00 GB"
