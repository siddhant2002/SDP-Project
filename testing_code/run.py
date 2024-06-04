import subprocess

# List of Python files to run
python_files = [
    "main.py",
    "face_top_30.py",
    "image_embedding_action_recognition1.py",
    "prophet_testing.py",
    "testing.py",
    "final_inference.py"
]

# Run each Python file
for file in python_files:
    subprocess.run(["python", file])
