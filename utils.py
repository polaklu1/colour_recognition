import os
import time

def get_ids_from_labels(labels):
    ids_list = []
    for item in labels:
        f_name_splitted = item['f_name'].split('_')
        ad_id = f_name_splitted[-2]

        if len(ad_id) > 7:
            ids_list.append(ad_id)

    ids_list = list(set(ids_list))

    return ids_list


def get_current_timestamp():
    return time.strftime("%Y_%m_%dT%H_%M")


def get_files_from_folder(path, only_jpg=True):
    files = []
    for r, d, f in os.walk(path):
        for fil in f:
            if only_jpg:
                if 'jpg' in fil:
                    files.append(os.path.join(r, fil))
            else:
                files.append(os.path.join(r, fil))
    return files



