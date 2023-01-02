import glob
import re

def get_img_segment_pairs(segment_data_path, img_data_path):

    segment_file_list = glob.glob(f'{segment_data_path}\*\*.png')
    img_file_list = glob.glob(f'{img_data_path}\*\*.png')

    exp_img = re.compile('\w+_\d{6}_\d{6}')
    exp_segment = re.compile('(\w+_\d{6}_\d{6})(_gtFine_labelIds)')
    segment_file_dict = {}
    img_file_dict = {}
    for t in segment_file_list:
        exp_result = exp_segment.search(t)
        if exp_result is not None:
            segment_file_dict[exp_result[1]] = t

    for t in img_file_list:
        img_file_dict[exp_img.findall(t)[-1]] = t

    img_segment_pair_list = []

    for file_name in segment_file_dict:
        if file_name in img_file_dict:
            img_segment_pair_list.append((img_file_dict[file_name], segment_file_dict[file_name]))
    return img_segment_pair_list


def get_img_detection_pairs(detection_file_path, img_data_path):

    detection_file_list = glob.glob(f'{detection_file_path}\*\*.txt')
    img_file_list = glob.glob(f'{img_data_path}\*\*.png')
    

    exp = re.compile('(\d{6}).(txt|png)')

    detection_file_dict = {}
    img_file_dict = {}
    img_detection_pair_list = []

    for t in detection_file_list:
        exp_result = exp.search(t)[1]
        if exp_result is not None:
            detection_file_dict[exp_result] = t

    for t in img_file_list:
        exp_result = exp.search(t)[1]
        if exp_result is not None:
            img_file_dict[exp_result] = t

    for t in detection_file_dict:
        if t in img_file_dict:
            img_detection_pair_list.append((img_file_dict[t], detection_file_dict[t]))

    return img_detection_pair_list

