import os
import sys
import shutil

VSLAM_LAB_PATH = os.getcwd()
sys.path.append(VSLAM_LAB_PATH)

from utilities import downloadFile

def glomap_setup():

    glomap_path = os.path.join(VSLAM_LAB_PATH, 'VSLAM-Baselines', 'glomap')
    glomap_extra_files = os.path.join(VSLAM_LAB_PATH, 'VSLAM-Baselines', 'extra-files', 'glomap')
    
    shutil.copytree(glomap_extra_files, glomap_path, dirs_exist_ok=True)

    # Download Vocabulary trees
    download_url = "https://demuc.de/colmap"
    vocabulary_trees = ["vocab_tree_flickr100K_words32K.bin",
                      "vocab_tree_flickr100K_words256K.bin",
                      "vocab_tree_flickr100K_words1M.bin"]

    for vocabulary_tree in vocabulary_trees:
      vocabulary_file = os.path.join(glomap_path, vocabulary_tree)
      if not os.path.exists(vocabulary_file):
          downloadFile(os.path.join(download_url, vocabulary_tree), glomap_path)


if __name__ == "__main__":
    glomap_setup()
