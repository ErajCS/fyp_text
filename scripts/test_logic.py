import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agri_ui'))

import drive_service as ds
from dotenv import load_dotenv

def test_full_logic():
    load_dotenv()
    
    # 1. Simulate app.py variables
    file_type = "image"
    category = "PQNK_research_and_knowledge_papers"
    original_name = "test_upload.jpg"
    
    # Create a dummy file
    local_temp = "dummy_test.txt"
    with open(local_temp, "w") as f:
        f.write("test content")
        
    type_folder_map = {"document": "PDFs", "image": "Images", "video": "Videos"}
    drive_path = [type_folder_map.get(file_type, "General"), category]
    
    print(f"Uploading to Drive Path: {drive_path}")
    
    result = ds.upload_file(
        local_path     = local_temp,
        filename       = original_name,
        mime_type      = "image/jpeg",
        subfolder_path = drive_path
    )
    
    print(f"Result: {result}")
    
    if os.path.exists(local_temp):
        os.remove(local_temp)

if __name__ == "__main__":
    test_full_logic()
