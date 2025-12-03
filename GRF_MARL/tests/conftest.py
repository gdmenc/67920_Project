"""
Shared test configuration and fixtures.
"""

import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestResults:
    """Track test results across modules."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
    
    def add(self, success, name=""):
        if success:
            self.passed += 1
        else:
            self.failed += 1
            if name:
                self.errors.append(name)
    
    def skip(self, name=""):
        self.skipped += 1
    
    @property
    def all_passed(self):
        return self.failed == 0
    
    def merge(self, other: 'TestResults'):
        """Merge results from another TestResults instance."""
        self.passed += other.passed
        self.failed += other.failed
        self.skipped += other.skipped
        self.errors.extend(other.errors)


def print_status(name, success, msg=""):
    """Print a test result with status indicator."""
    status = "✅" if success else "❌"
    print(f"  {status} {name}" + (f": {msg}" if msg else ""))
    return success


def print_section(title):
    """Print a section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n  [{title}]")


# Standard test configurations
STANDARD_MODEL_CONFIG = {
    "model": "gr_football.hierarchical",
    "initialization": {
        "use_orthogonal": True,
        "gain": 1.0,
    },
    "actor": {
        "network": "mlp",
        "layers": [
            {"units": 256, "activation": "ReLU"},
            {"units": 128, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
    "critic": {
        "network": "mlp",
        "layers": [
            {"units": 256, "activation": "ReLU"},
            {"units": 128, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
}

STANDARD_CUSTOM_CONFIG = {
    "FE_cfg": {"num_players": 10},  # 5v5
    "use_cuda": False,
    "use_rnn": False,
    "rnn_layer_num": 1,
    "use_feature_normalization": True,
    "use_popart": False,
    "use_q_head": False,
}

SUB_POLICY_CONFIGS = [
    {"name": "defense_v3", "path": "light_malib/trained_models/gr_football/5_vs_5/defense_v3"},
    {"name": "PassingMain_v2", "path": "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2"},
    {"name": "3-1_formation", "path": "light_malib/trained_models/gr_football/5_vs_5/3-1_formation"},
    {"name": "3-1_LongPass", "path": "light_malib/trained_models/gr_football/5_vs_5/3-1_LongPass"},
]

