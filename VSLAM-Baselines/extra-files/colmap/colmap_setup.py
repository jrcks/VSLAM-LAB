import os
import sys
import shutil

VSLAM_LAB_PATH = os.getcwd()
sys.path.append(VSLAM_LAB_PATH)

from utilities import downloadFile

def colmap_setup():

    colmap_path = os.path.join(VSLAM_LAB_PATH, 'VSLAM-Baselines', 'colmap')
    colmap_extra_files = os.path.join(VSLAM_LAB_PATH, 'VSLAM-Baselines', 'extra-files', 'colmap')
    
    shutil.copytree(colmap_extra_files, colmap_path, dirs_exist_ok=True)

    # Download Vocabulary trees
    download_url = "https://demuc.de/colmap"
    vocabulary_trees = ["vocab_tree_flickr100K_words32K.bin",
                       "vocab_tree_flickr100K_words256K.bin",
                       "vocab_tree_flickr100K_words1M.bin"]

    for vocabulary_tree in vocabulary_trees:
       vocabulary_file = os.path.join(colmap_path, vocabulary_tree)
       if not os.path.exists(vocabulary_file):
           downloadFile(os.path.join(download_url, vocabulary_tree), colmap_path)


if __name__ == "__main__":
    colmap_setup()
