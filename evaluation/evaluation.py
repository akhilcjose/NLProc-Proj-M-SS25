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

for entry in test_data:
    print(f"Time: {entry['timestamp']}")
    print(f"Question: {entry['question']}")
    print(f"Answer: {entry['generated_answer']}")
    print(f"group_id:{entry['group_id']}")
    print(f"retrieved_chunks:{entry['retrieved_chunks']}")
    print()