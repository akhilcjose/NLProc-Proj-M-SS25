import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, '..', 'test_inputs.json')

print("Current script dir:", current_dir)
print("Looking for JSON file at:", json_path)

# List files in the directory where it expects JSON
print("Files in JSON folder:", os.listdir(os.path.dirname(json_path)))

with open(json_path, 'r') as f:
    test_data = json.load(f)

for test in test_data:
    print("Question:", test['question'])
    print("Expected answer:", test['expected_answer'])
    print()