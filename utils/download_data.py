
import os
import zipfile
import requests
from pathlib import Path

def download_data():
  data_path = Path("data")
  image_path = data_path/"league_dataset"

  if not image_path.is_dir():
    image_path.mkdir(parents=True,exist_ok = True)

    with open(data_path/"league_dataset.zip","wb") as f:
      request = requests.get("https://github.com/HatemSal/LOL_Champ_Classifier/raw/refs/heads/main/league_dataset.zip")
      f.write(request.content)

    with zipfile.ZipFile(data_path/"league_dataset.zip",'r') as zip_ref:
      zip_ref.extractall(image_path)
  dataset_path = image_path/"league_dataset"
  return dataset_path
