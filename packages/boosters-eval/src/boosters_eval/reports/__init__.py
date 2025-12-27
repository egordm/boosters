"""Report formatters for benchmark results.

This module provides unified formatters for different output types:
- console: Rich table output for terminal
- markdown: Markdown report generation
"""

from boosters_eval.reports.base import (
    LibraryVersions,
    MachineInfo,
    ReportMetadata,
    get_git_sha,
    get_library_versions,
    get_machine_info,
    is_significant,
)
from boosters_eval.reports.console import format_results_terminal
from boosters_eval.reports.markdown import generate_report, render_report

__all__ = [
    "LibraryVersions",
    "MachineInfo",
    "ReportMetadata",
    "format_results_terminal",
    "generate_report",
    "get_git_sha",
    "get_library_versions",
    "get_machine_info",
    "is_significant",
    "render_report",
]
