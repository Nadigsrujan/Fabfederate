import sys

from fabfederate.cli import main

main(["audit", *sys.argv[1:]])
