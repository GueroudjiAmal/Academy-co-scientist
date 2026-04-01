# academy_coscientist/plugins/__init__.py
# Re-export from the flowcept agentic package — local plugin files kept for reference.
from flowcept.agents.academy.academy_plugin import FlowceptAcademyPlugin
from flowcept.agents.academy.academy_plugin import FlowceptAcademyPlugin as FlowceptPlugin

__all__ = [
    "FlowceptAcademyPlugin",
    "FlowceptPlugin",
]
