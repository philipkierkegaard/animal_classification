import os

# Define paths dynamically based on the typical cookiecutter structure

# _TEST_ROOT points to the root of the tests folder
_TEST_ROOT = os.path.dirname(os.path.abspath(__file__))  # The directory containing this test file

# _PROJECT_ROOT points to the root of the entire project
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_ROOT, os.pardir))  # Moves up one level from tests

# _PATH_DATA points to the data folder inside the project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # Assumes a 'data' folder at the root of the project

# Optional: Print paths for debugging
print("_TEST_ROOT:", _TEST_ROOT)
print("_PROJECT_ROOT:", _PROJECT_ROOT)
print("_PATH_DATA:", _PATH_DATA)
