import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agri_ui'))

import drive_service as ds
from dotenv import load_dotenv

def diag():
    load_dotenv()
    root_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    print(f"Root: {root_id}")
    
    # 1. List Root
    items = ds.list_folder_contents(root_id)
    print("Items in Root:")
    found_images = False
    for item in items:
        print(f" - {item['name']} ({item['id']}) - {item['mimeType']}")
        if item['name'] == 'Images':
            found_images = item['id']
            
    if found_images:
        print(f"Images folder found ({found_images})")
        # 2. List Images
        subitems = ds.list_folder_contents(found_images)
        print("Items in Images:")
        for sub in subitems:
            print(f"   - {sub['name']} ({sub['id']})")
    else:
        print("Images folder NOT FOUND in root.")

if __name__ == "__main__":
    diag()
