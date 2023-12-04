import glob
import json
import ntpath
import os
import sys


# TOKEN_BOS = '<|endoftext|>'
# TOKEN_EOS = '<|END|>'
# TOKEN_PAD = '<|pad|>'
# TOKEN_SEP = '\n\n####\n\n'


def read_file(file_path, encoding):
    with open(file_path, 'r', encoding=encoding) as f:
        '''
        content = f.read()
        sentence = content.split('\n')  # 根據換行字符分割成句子
        print('content:', '\n'+content)
        '''
        sentence = f.readlines()
        # print('sentence:', sentence)
    return sentence


def process_annotation_file(lines):
    entity_dict = {}  # entity_dict[10, 11, 31, 87, 9]
    # print('lines:', lines)
    for index, line in enumerate(lines):  # 遍歷 lines 中的每一行, lines 是 answer_70m_epoch_3_batch_10_lr5e_5.txt 中的字串
        # print('line:', line)
        items = line.strip('\n').split('\t')  # 將字符串 line 去除換行符 \n，然後按照制表符 \t 進行分割，返回一個由分割後的子字符串組成的列表
        # print('items:', items)
        # items = [10, DOCTOR, 1337, 1339, IU]
        # items = [10, DATE	1346, 1354, 27.10.15, 2015-10-27]

        if len(items) == 5:
            item_dict = {
                'file_id': items[0],
                'phi': items[1],
                'start_index': int(items[2]),
                'end_index': int(items[3]),
                'label': items[4]

            }
        elif len(items) == 6:
            item_dict = {
                'file_id': items[0],
                'phi': items[1],
                'start_index': int(items[2]),
                'end_index': int(items[3]),
                'label': items[4],
                'normalize_time': items[5]
            }
        else:
            continue
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
        # entity_dict = {file1: [item_dict_line1, item_dict_line2, ...], file2: [...], ...}
        # where item_dict_line1 = {'phi': 'IDNUM','st_idx': 14,'ed_idx': 24,'entity': '38R302659X'}
        # file1 = '10'
    return entity_dict


def generate_annotated_medical_report(file_path):
    anno_lines = read_file(file_path, "utf-8-sig")
    annos_dict = process_annotation_file(anno_lines)
    return annos_dict


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def process_medical_report(filename, dataset_file, annos_dict, delimiter):
    # 讀取病理報告

    sents = read_file(dataset_file, "utf-8")
    article = "".join(sents)  # 將一個列表中的所有字符串元素連接成一個字符串
    boundary, item_index, temp_seq, seq_pairs = 0, 0, "", []
    temp_array = []
    line_counter = 0

    for word_index, word in enumerate(article):  # 字元位置, 字元 in enumerate(此病理報告每行串接而成的字串)
        item_start_index = annos_dict[filename][item_index]['start_index']

        if word == '\n':
            line_counter = line_counter + 1

        if word_index == item_start_index:  # 如果字元的位置與 annotation 中，第 item_idx 行 的 PHI 起始位置相同
            phi_key = annos_dict[filename][item_index]['phi']  # phi_key = 第 item_idx 行的 phi 種類
            phi_value = annos_dict[filename][item_index]['label']  # phi_value = 第 item_idx 行的 entity 字串

            if 'normalize_time' in annos_dict[filename][item_index]:
                temp_dict = {
                    'line_index': line_counter,
                    'phi': phi_key,
                    'answer': phi_value,
                    'normalize_time': annos_dict[filename][item_index]['normalize_time']
                }
            else:
                temp_dict = {
                    'line_index': line_counter,
                    'phi': phi_key,
                    'answer': phi_value}

            newline_index = article.find('\n', word_index, len(article))
            previous_line_index = article.rfind('\n', 0, word_index)

            if previous_line_index == -1:
                previous_line_index = 0

            temp_array.append(temp_dict)

            if newline_index == -1:
                pass

            if item_index == len(annos_dict[filename]) - 1:
                continue
            item_index += 1  # 已執行過的行數+1

    line_map = {}
    for index, line_str in enumerate(sents):
        kkk = []
        for item in temp_array:
            line_index = item['line_index']
            if line_index == index:
                kkk.append(item)
        # if len(kkk) > 0:
            line_map[index] = kkk

    outputs = []
    for key in line_map.keys():
        items = line_map[key]
        for item in items:
            del item['line_index']
        outputs.append(f"{json.dumps(items)}{delimiter}{sents[key]}")
    # temp_seq = (
    #     f"{}{delimiter}{line_counter}{delimiter}{phi_key}{delimiter}{item_start_index}{delimiter}{item_end_index}{delimiter}{phi_value}"
    #     f"{delimiter}{annos_dict[filename][item_idx]['normalize_time']}")
    return outputs


if __name__ == '__main__':
    args = sys.argv
    print(args)
    dataset_folder = args[1]
    answer_file = args[2]
    tsv_save_path = args[3]
    dataset_files = glob.glob(dataset_folder)

    annotated = generate_annotated_medical_report(answer_file)
    results = []
    for file in dataset_files:
        filename = os.path.splitext(path_leaf(file))[0]
        delimiter = bytes([0x04, 0x04])
        results.append(process_medical_report(filename, file, annotated, delimiter.decode('utf-8')))

    with open(tsv_save_path, 'w', encoding='utf-8') as f:
        for lines in results:
            f.writelines(lines)
