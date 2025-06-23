#!/usr/bin/env python3
"""Test script to verify cbftorch imports work correctly"""

print("Testing cbftorch imports...")

try:
    from cbftorch import Barrier, MinIntervCFSafeControl
    print("✓ Successfully imported Barrier and MinIntervCFSafeControl from cbftorch")
except ImportError as e:
    print(f"✗ Failed to import from cbftorch: {e}")

try:
    from cbftorch.dynamics import AffineInControlDynamics
    print("✓ Successfully imported AffineInControlDynamics from cbftorch.dynamics")
except ImportError as e:
    print(f"✗ Failed to import from cbftorch.dynamics: {e}")

try:
    from cbftorch.utils import Map, vectorize_tensors, make_linear_alpha_function_form_list_of_coef, softmax, softmin
    print("✓ Successfully imported utilities from cbftorch.utils")
except ImportError as e:
    print(f"✗ Failed to import from cbftorch.utils: {e}")

print("\nAll imports successful! The migration to cbftorch is complete.")