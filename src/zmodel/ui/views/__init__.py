"""UI views for Z-ModelPilot."""

from zmodel.ui.views.benchmark import render_benchmark_view
from zmodel.ui.views.compatibility import render_compatibility_view
from zmodel.ui.views.graph_optimizer import render_graph_optimization_view
from zmodel.ui.views.inspector import render_inspector_view
from zmodel.ui.views.optimization import render_optimization_view
from zmodel.ui.views.overview import render_overview_view
from zmodel.ui.views.upcoming import render_upcoming_view
from zmodel.ui.views.validation import render_validation_view
from zmodel.ui.views.workers import render_workers_view

__all__ = [
    "render_overview_view",
    "render_inspector_view",
    "render_compatibility_view",
    "render_benchmark_view",
    "render_optimization_view",
    "render_graph_optimization_view",
    "render_workers_view",
    "render_validation_view",
    "render_upcoming_view",
]

