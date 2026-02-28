import os
import shutil

# Define the root path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The directories we want to completely wipe
FOLDERS_TO_NUKE = [
    os.path.join(BASE_DIR, "training", "logs"),
    os.path.join(BASE_DIR, "training", "models"),
    os.path.join(BASE_DIR, "client", "public", "model"),
    os.path.join(BASE_DIR, "data", "processed") # Wipes old 15-word extractions
]

def clean_slate():
    print("Initiating Tacit Clean Slate Protocol...\n")
    
    for folder in FOLDERS_TO_NUKE:
        if os.path.exists(folder):
            try:
                # Delete the folder and all its contents
                shutil.rmtree(folder)
                print(f"✅ DELETED: {os.path.relpath(folder, BASE_DIR)}")
                # Recreate the empty folder so scripts don't crash looking for it
                os.makedirs(folder)
            except Exception as e:
                print(f"❌ ERROR deleting {folder}: {e}")
        else:
            # If it doesn't exist, just create it fresh
            os.makedirs(folder)
            print(f"✨ CREATED FRESH: {os.path.relpath(folder, BASE_DIR)}")

    print("\n Clean complete")

if __name__ == "__main__":
    clean_slate()