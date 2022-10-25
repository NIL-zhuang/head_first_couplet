import os

WORKING_DIR = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__),
        "../..")
)
COUPLET_PATH = os.path.join(WORKING_DIR, "data", "couplet")
