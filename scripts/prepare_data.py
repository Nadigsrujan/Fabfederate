"""Compatibility entry point; run from an installed project environment."""
import sys

from fabfederate.cli import main

main(["prepare", *sys.argv[1:]])
