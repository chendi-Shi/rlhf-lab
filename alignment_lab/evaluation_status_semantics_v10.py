"""Strict V9-to-V10 execution AST proof for the isolated status-I/O repair.

This validator does not execute either source. Every permitted insertion has
an exact AST template; after removing only those insertions, the whole module
must equal its frozen V9 predecessor. Model, input, metric and manifest logic
are included in that comparison.
"""
from __future__ import annotations

import ast
import copy


ADDED_SOURCES = (
    "alignment_lab/evaluation_resources_v10.py", "tests_v2/test_evaluation_resources_v10.py",
    "alignment_lab/evaluation_execution_v10.py", "tests_v2/test_evaluation_execution_v10.py",
    "scripts/run_alignlab_evaluation_stage_v10.py",
    "alignment_lab/evaluation_capacity_inheritance_v10.py", "tests_v2/test_evaluation_capacity_inheritance_v10.py",
    "alignment_lab/evaluation_atomic_status_v10.py", "tests_v2/test_evaluation_atomic_status_v10.py",
    "alignment_lab/evaluation_status_semantics_v10.py", "tests_v2/test_evaluation_status_semantics_v10.py",
)

PROGRESS = '''
def _progress(publisher, value, *, terminal=False):
    return publisher.publish(value, terminal=terminal)
'''
SECONDARY = '''
def _secondary_failure(value, primary, phase, error):
    """Keep cleanup/publication diagnostics without replacing the first error."""
    detail = {"phase": phase, "error_type": type(error).__name__, "error": str(error)}
    if value is not None:
        value.setdefault("secondary_errors", []).append(detail)
    if primary is not None:
        primary.add_note(f"{phase}: {type(error).__name__}: {error}")
'''
PUBLISHER = '''
publisher = AtomicStatusPublisher(output / "progress.json",
    deadline_epoch=request["deadline_epoch"], serialize=canonical,
    stop_requested=lambda: (output / "STOP").exists() or (output.parent / "STOP").exists(),
    now_fn=system.now, sleep_fn=system.sleep)
'''
TERMINAL = '''
try:
    _progress(publisher, status, terminal=True)
except BaseException as publication_error:
    _secondary_failure(status, error, "terminal_status_publication", publication_error)
    status["status_publication_failure"] = publisher.failure
    error.add_note(f"Status publication evidence: {publisher.failure!r}")
'''
CLEANUP = '''
try:
    if lock.is_file() and parse(lock.read_bytes()) == owner:
        lock.unlink()
except BaseException as cleanup_error:
    if primary_error is None:
        raise
    _secondary_failure(status, primary_error, "exact_owner_lock_cleanup", cleanup_error)
'''


def _dump(value):
    return ast.dump(value, include_attributes=False)


def _statement(source):
    tree = ast.parse(source)
    if len(tree.body) != 1:
        raise ValueError("One exact status-repair AST statement required")
    return tree.body[0]


def _same(actual, expected, label):
    if _dump(actual) != _dump(expected):
        raise ValueError("V10 status repair AST differs: " + label)


def _function(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Exactly one execution function required: " + name)
    return matches[0]


def _assignment(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == name]
    if len(matches) != 1:
        raise ValueError("Exactly one execution assignment required: " + name)
    return matches[0]


def validate_v10_execution_semantics(prior_source, current_source):
    """Require a fully equal module after exactly enumerated V10 I/O changes.

    Callers must verify the bytes against their frozen plan pins before calling.
    The predecessor's own V6-to-V9 proof remains a separate required check.
    """
    previous, current = ast.parse(prior_source), ast.parse(current_source)
    normalized = copy.deepcopy(current)
    atomic_import = _statement("from .evaluation_atomic_status_v10 import AtomicStatusPublisher")
    matches = [node for node in normalized.body if _dump(node) == _dump(atomic_import)]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen atomic status publisher import required")
    normalized.body.remove(matches[0])

    before = _assignment(previous, "REQUIRED_SOURCES")
    after = _assignment(normalized, "REQUIRED_SOURCES")
    if (not isinstance(before.value, ast.Call) or not isinstance(after.value, ast.Call)
            or len(before.value.args) != 1 or len(after.value.args) != 1
            or not isinstance(before.value.args[0], ast.Set)
            or not isinstance(after.value.args[0], ast.Set)):
        raise ValueError("Literal complete frozen execution source inventory required")
    expected = copy.deepcopy(before)
    expected.value.args[0].elts.extend(ast.Constant(name) for name in ADDED_SOURCES)
    _same(after, expected, "append all prior source pins and exact V10 dependencies")
    after.value = copy.deepcopy(before.value)

    progress = _function(normalized, "_progress")
    _same(progress, _statement(PROGRESS), "canonical publisher progress adapter")
    normalized.body[normalized.body.index(progress)] = copy.deepcopy(_function(previous, "_progress"))
    secondary = _function(normalized, "_secondary_failure")
    _same(secondary, _statement(SECONDARY), "secondary error diagnostic helper")
    normalized.body.remove(secondary)

    run = _function(normalized, "run_evaluation_stage")
    publisher = _assignment(run, "publisher")
    _same(publisher, _statement(PUBLISHER), "one publisher with original deadline and STOP")
    owner_index = next((index for index, node in enumerate(run.body)
                        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                           and target.id == "owner" for target in node.targets)), -1)
    if owner_index < 0 or run.body.index(publisher) != owner_index + 1:
        raise ValueError("Publisher must be constructed after owner identity and before GPU lock")
    run.body.remove(publisher)
    initial = [node for node in run.body if _dump(node) == _dump(
        _statement("status, guard, primary_error = None, None, None"))]
    if len(initial) != 1:
        raise ValueError("One first-error latch required before the stage try")
    run.body[run.body.index(initial[0])] = _statement("status, guard = None, None")
    tries = [node for node in run.body if isinstance(node, ast.Try)]
    if len(tries) != 1 or len(tries[0].handlers) != 1:
        raise ValueError("One original full stage lifecycle try required")
    lifecycle = tries[0]
    handler = lifecycle.handlers[0]
    if not handler.body:
        raise ValueError("First exception preservation is missing")
    _same(handler.body[0], _statement("primary_error = error"), "first exception latch")
    handler.body.pop(0)
    if len(handler.body) != 2 or not isinstance(handler.body[0], ast.If):
        raise ValueError("Original status-error and bare-raise structure required")
    status_handler = handler.body[0]
    if not status_handler.body:
        raise ValueError("Original status exception evidence is missing")
    _same(status_handler.body[-1], _statement(TERMINAL), "terminal publication preserves first error")
    status_handler.body[-1] = _statement('_progress(output / "progress.json", status)')
    if len(lifecycle.finalbody) != 1:
        raise ValueError("One exact-owned cleanup block required")
    _same(lifecycle.finalbody[0], _statement(CLEANUP), "exact-owned cleanup always runs")
    lifecycle.finalbody = copy.deepcopy(_statement(CLEANUP).body)

    class RoutingAndProgress(ast.NodeTransformer):
        def __init__(self):
            self.progress_calls = 0
            self.resource_imports = 0
            self.capacity_imports = 0

        def visit_ImportFrom(self, node):
            if node.level == 1 and node.module == "evaluation_resources_v10":
                node.module = "evaluation_resources_v9"
                self.resource_imports += 1
            elif node.level == 1 and node.module == "evaluation_capacity_inheritance_v10":
                node.module = "evaluation_capacity_inheritance_v9"
                self.capacity_imports += 1
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for name in ("run_alignlab_evaluation_stage", "supervise_alignlab_evaluation"):
                    node.value = node.value.replace(name + "_v10.py", name + "_v9.py")
            return node

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == "_progress"
                    and node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "publisher"):
                _same(node, _statement("_progress(publisher, status)").value,
                      "normal progress publication call")
                node.args[0] = _statement('output / "progress.json"').value
                self.progress_calls += 1
            return self.generic_visit(node)

    routing = RoutingAndProgress()
    normalized = routing.visit(normalized)
    if (routing.progress_calls, routing.resource_imports, routing.capacity_imports) != (3, 1, 1):
        raise ValueError("Exact three progress sites and two versioned dependencies required")
    _same(normalized, previous, "complete execution module including model/data/manifest semantics")
    return {"status": "v10_execution_status_io_only", "full_module_ast_equal_after_exact_normalization": True,
            "model_or_tensor_imported": False, "progress_publication_sites": 3}
