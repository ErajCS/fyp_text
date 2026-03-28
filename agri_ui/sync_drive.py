"""
sync_drive.py
Populate the database by scanning the Google Drive structure:
Root > [PDFs, Images, Videos] > [Category Folders] > [Files]
"""

import os
import sys
import logging

# Ensure we can import app and drive_service
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Resource, User
import drive_service as ds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync():
    root_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not root_id:
        logger.error("GOOGLE_DRIVE_FOLDER_ID not set in .env")
        return

    # 1. Get a system user to assign as uploader (fallback to first admin)
    with app.app_context():
        admin = User.query.filter(User.role.in_(['admin', 'superadmin'])).first()
        admin_id = admin.id if admin else 1

        logger.info(f"Scanning Root Folder: {root_id}")
        top_folders = ds.list_folder_contents(root_id)
        
        # We look for "PDFs", "Images", "Videos" (case insensitive)
        type_map = {
            "pdfs":   "document",
            "images": "image",
            "videos": "video"
        }

        for top in top_folders:
            name_lower = top['name'].lower()
            if top['mimeType'] != 'application/vnd.google-apps.folder':
                continue
            
            if name_lower not in type_map:
                logger.info(f"Skipping unknown top-level folder: {top['name']}")
                continue
            
            file_type = type_map[name_lower]
            logger.info(f"--- Processing {top['name']} ({file_type}) ---")
            
            # 2. Process Category Subfolders (e.g. water_management)
            categories = ds.list_folder_contents(top['id'])
            for cat in categories:
                if cat['mimeType'] == 'application/vnd.google-apps.folder':
                    category_name = cat['name']
                    logger.info(f"  Category: {category_name}")
                    
                    # 3. Process Files inside Category
                    files = ds.list_folder_contents(cat['id'])
                    for f in files:
                        if f['mimeType'] == 'application/vnd.google-apps.folder':
                            continue
                        
                        # Check if already exists
                        existing = Resource.query.filter_by(drive_file_id=f['id']).first()
                        if existing:
                            logger.info(f"    Already synced: {f['name']}")
                            continue
                        
                        # Add to DB
                        new_res = Resource(
                            title         = f['name'].rsplit('.', 1)[0],
                            description   = f"Imported from Drive: {category_name}",
                            category      = category_name,
                            keywords      = category_name.replace('_', ' '),
                            file_type     = file_type,
                            filename      = None, # No local copy
                            original_name = f['name'],
                            video_link    = None, # Sync only Drive videos for now
                            drive_file_id = f['id'],
                            drive_view_link = f.get('webViewLink'),
                            uploaded_by   = admin_id
                        )
                        db.session.add(new_res)
                        logger.info(f"    Added: {f['name']}")
                
                else:
                    # File directly in top level (e.g. PDFs folder but no subfolder)
                    f = cat
                    existing = Resource.query.filter_by(drive_file_id=f['id']).first()
                    if not existing:
                        new_res = Resource(
                            title         = f['name'].rsplit('.', 1)[0],
                            description   = f"Imported from Drive",
                            category      = "General",
                            file_type     = file_type,
                            drive_file_id = f['id'],
                            drive_view_link = f.get('webViewLink'),
                            uploaded_by   = admin_id
                        )
                        db.session.add(new_res)
                        logger.info(f"    Added root file: {f['name']}")

        db.session.commit()
        logger.info("✅ Sync Complete!")

if __name__ == "__main__":
    sync()
