import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

apiKey = os.getenv('isaac_apiKey')
print("API key loaded:", apiKey)

url = "https://api.themoviedb.org/3/discover/movie"
headers = {
    "Authorization": f"Bearer {apiKey}",
    "accept": "application/json"
}

movie_ids = set()

for year in range(2005, 2025):
    params = {
        "primary_release_date.gte": f"{year}-01-01",
        "primary_release_date.lte": f"{year}-12-31",
        "with_release_type": "3",  # theatrical releases
        "page": 1
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    tp = data.get('total_pages', 1)

    # First page
    for movie in data.get("results", []):
        movie_ids.add(movie.get("id"))

    # Remaining pages
    for i in tqdm(range(2, int(tp) + 1), desc=f"Year {year}"):
        params["page"] = i
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        for movie in data.get("results", []):
            movie_ids.add(movie.get("id"))

        time.sleep(0.25)

print(f"Fetched {len(movie_ids)} movies")

with open("ids.txt", "w") as f:
    for id in movie_ids:
        f.write(f"{id}\n")




# params = {
#     "primary_release_date.gte": "2024-01-01",  # start date
#     "primary_release_date.lte": "2024-12-31",  # end date
#     "with_release_type": "3",  # theatrical releases
#     "page": 1
# }

# movie_ids = set()

# #get inital page count
# response = requests.get(url, headers=headers, params=params)
# data = response.json()
# tp = data.get('total_pages')

# for movie in data.get("results", []):
#         movie_ids.add(movie.get("id"))

# for i in tqdm(range(2, int(tp))):
#     params["page"] = i
#     response = requests.get(url, headers=headers, params=params)
#     data = response.json()

#     # Add current page results
#     for movie in data.get("results", []):
#         movie_ids.add(movie.get("id"))

#     time.sleep(.25)

# print(f"Fetched {len(movie_ids)} movies")


# with open("ids.txt", "w") as f:
#     for id in movie_ids:
#         f.write(f"{id}\n")
