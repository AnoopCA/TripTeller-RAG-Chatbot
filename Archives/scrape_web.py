import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Munnar tourism page
url = "https://munnar.com/"
# url = "https://www.tripuntold.com/kerala/idukki/munnar/"
# url = "https://www.keralatourism.org/ebrochures/munnar-mesmerisingly-yours/59/"
# url = "https://en.wikipedia.org/wiki/Munnar"
# url = "https://www.makemytrip.com/tripideas/places/munnar"
# url = "https://gostops.com/blog/all-about-munnar-the-kashmir-of-south-india/"
# url = "https://idukki.nic.in/en/tourist-place/munnar/"
# url = "https://cloudcastlemunnar.com/about-munnar/"

response = requests.get(url)

# Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")

# Extract destination title
title = soup.find("h1").get_text(strip=True)

# Extract short description
description = soup.find("div", class_="content_text").get_text(" ", strip=True)

# Extract nearby attractions (if available)
attractions = []
for item in soup.find_all("div", class_="list_details"):
    name = item.find("h3").get_text(strip=True) if item.find("h3") else None
    desc = item.find("p").get_text(strip=True) if item.find("p") else None
    if name:
        attractions.append({"Name": name, "Description": desc})

# Save as dataset
df = pd.DataFrame(attractions)
df.to_csv("munnar_attractions.csv", index=False)

print("Scraped", len(attractions), "attractions for Munnar.")
