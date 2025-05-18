import json
import time
from datetime import datetime

LOG_FILE = 'query_logs.jsonl'

def log_query(question, retrieved_chunks, prompt, generated_answer, group_id=None):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'question': question,
        'retrieved_chunks': retrieved_chunks,  # ideally list of text or IDs
        'prompt': prompt,
        'generated_answer': generated_answer,
        'group_id': group_id,
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')