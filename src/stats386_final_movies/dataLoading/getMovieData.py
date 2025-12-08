import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json

load_dotenv()
apiKey = os.getenv('isaac_apiKey')

ids = None
with open('ids.txt', 'r') as f:
    ids = [line.strip() for line in f]


# Check progress
progress_file = "progress.txt"
start_index = 0
if os.path.exists(progress_file):
    with open(progress_file, 'r') as pf:
        start_index = int(pf.read().strip())

print(f"Resuming from index {start_index} (ID: {ids[start_index]})")

# If resuming, load existing data
if os.path.exists("movies.json"):
    with open("movies.json", "r") as f:
        all_data = json.load(f)

# Load existing data if any
all_data = []
if os.path.exists("movies.json"):
    with open("movies.json", "r") as f:
        all_data = json.load(f)

for i in tqdm(range(start_index, len(ids))):
    id = ids[i]
    url = f"https://api.themoviedb.org/3/movie/{id}"
    headers = {
        "Authorization": f"Bearer {apiKey}",
        "accept": "application/json"
    }

    try:
        response = requests.get(url=url, headers=headers, timeout=10)
        data = response.json()
        all_data.append(data)

        # Save progress
        with open(progress_file, 'w') as pf:
            pf.write(str(i + 1))

        # Save periodically
        if (i + 1) % 50 == 0:
            with open("movies.json", "w") as f:
                json.dump(all_data, f)

        time.sleep(0.25)

    except Exception as e:
        print(f"Error on ID {id}: {e}")
        continue  # instead of break

# Final save
with open("movies.json", "w") as f:
    json.dump(all_data, f)


print("Done!")

