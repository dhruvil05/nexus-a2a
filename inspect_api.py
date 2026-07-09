"""
Run this ONCE from your project root to print real nexus_a2a API signatures.
Paste output back to Claude so tests match exactly.

Usage:
    python inspect_api.py
"""

import importlib
import inspect

MODULES = [
    "nexus_a2a.core.orchestrator",
    "nexus_a2a.core.registry",
    "nexus_a2a.core.dead_letter",
    "nexus_a2a.transport.http_client",
    "nexus_a2a.models.task",
]

for mod_name in MODULES:
    try:
        mod = importlib.import_module(mod_name)
        print(f"\n{'=' * 60}")
        print(f"MODULE: {mod_name}")
        print("=" * 60)
        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod_name:
                continue
            print(f"\n  class {cls_name}")
            # __init__
            try:
                print(f"    __init__{inspect.signature(cls.__init__)}")
            except Exception:
                pass
            # public methods
            for mname, mobj in inspect.getmembers(cls, predicate=inspect.isfunction):
                if mname.startswith("_"):
                    continue
                try:
                    print(f"    def {mname}{inspect.signature(mobj)}")
                except Exception:
                    print(f"    def {mname}(?)")
    except ImportError as e:
        print(f"\nCANNOT IMPORT {mod_name}: {e}")
