import os
import re
import shutil
import zipfile


def extract_file_id(fname: str) -> int:

    print("Extracting id from " + fname)
    return int(re.search('\d+', fname).group())


def unzip(archive_name: str, extract_path: str) -> None:

    zip_ref = zipfile.ZipFile(archive_name, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()


def cp_n_files(src_dir, dst_dir, n):
    files = os.listdir(src_dir)[:n]
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
