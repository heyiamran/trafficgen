import zipfile

with zipfile.ZipFile("/content/drive/MyDrive/Research/Traffic/Data/processed.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/drive/MyDrive/Research/Traffic/Data/")