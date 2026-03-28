import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agri_ui'))

from app import app, db, Resource

def debug():
    with app.app_context():
        res = Resource.query.order_by(Resource.id.desc()).limit(5).all()
        print("-" * 50)
        for r in res:
            print(f"ID: {r.id}")
            print(f"Title: {r.title}")
            print(f"Drive ID: {r.drive_file_id}")
            print(f"View Link: {r.drive_view_link}")
            print(f"Filename: {r.filename}")
            print(f"Category: {r.category}")
            print("-" * 50)

if __name__ == "__main__":
    debug()
