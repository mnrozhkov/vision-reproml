import re


def extract_file_id(fname):

    print("Extracting id from " + fname)
    return int(re.search('\d+', fname).group())
