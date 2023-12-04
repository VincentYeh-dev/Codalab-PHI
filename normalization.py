import os
import glob
import json
import re


def clear_file(path):
    with open(path, 'w', encoding='utf-8') as _file:
        _file.truncate(0)


def read_file2text(path):
    with open(path, 'r', encoding='utf-8') as _file:
        file_content = _file.read()
        return file_content


def keyword_groups_divider(content):
    # print('content:', content)
    # matches = re.findall(r'["phi"].*[^["phi"]]*', content)
    matches = re.findall(r'("phi".*?)(?="phi"|$)', content)

    parsed_dicts = []
    for match in matches:
        # print(match)
        parsed_dicts.append(match)
    return parsed_dicts


def contains_alphanumeric(text):
    # 檢查字符串中是否包含 A-Z、a-z 或 0-9 中的字符
    return re.search(r'[A-Za-z0-9]', text) is not None


def keyword_classif(content):
    # print('gen_content:', content)
    matches = re.findall(r'"(phi|answer|normalize_time)".*?"(.*?)"', content)
    # matches = [('phi', 'DATE'), ('answer', '1/1/1991'), ('normalize_time', '1991-01-01')]

    if all(_key in [match[0] for match in matches] for _key in ['phi', 'answer']):    # 是否含有 'phi' 與 'answer' 項目
        result_dict = {}
        for match in matches:
            _key, value = match
            if not contains_alphanumeric(value):
                return None
            result_dict[_key] = value
        return result_dict
    else:
        return None


def get_newline_pos(content, line_number):
    lines = content.split('\n')
    if line_number <= 0 or line_number > len(lines):
        return None, None

    start_position = 0
    for i in range(1, line_number):     # 由第一行算到第 line_number 行
        start_position += len(lines[i - 1]) + 1  # start_position += 該行字符數+換行符
    # print('lines:', start_position, lines[line_number])
    return start_position, lines[line_number - 1]


def find_keyword_pos(content, keyword, offset=0):
    # print("finding:", re.escape(keyword))
    # 定義正則表達式模式，查找 keyword 緊鄰 A-Z, a-z, 0-9 之外的字符
    pattern = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
    matches = re.finditer(pattern, content)     # 使用正則表達式查找匹配項
    # print("num of matches:", len(list(matches)))

    pos = []
    for match in matches:
        start = match.start()+offset
        end = match.end()+offset
        pos.append((start, end))
    return pos


def to_text_list(keyword_dict, file_name):
    if not keyword_dict["pos"]:
        return None
    output = []
    if keyword_dict.get("normalize_time", ""):
        for pos in keyword_dict["pos"]:
            line = "{}\t{}\t{}\t{}\t{}\t{}".format(file_name, keyword_dict["phi"], pos[0], pos[1],
                                                   keyword_dict["answer"], keyword_dict["normalize_time"])
            output.append(line)
    else:
        for pos in keyword_dict["pos"]:
            line = "{}\t{}\t{}\t{}\t{}".format(file_name, keyword_dict["phi"], pos[0], pos[1],
                                               keyword_dict["answer"])
            output.append(line)
        # print(output)
    return output


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def gen_text_normalization(file_name, ori_gen_content, ori_file_content, line_number=None):
    ori_gen_content_list = keyword_groups_divider(ori_gen_content)
    text_list = []
    for single_feature_gen_content in ori_gen_content_list:
        single_feature_dict = keyword_classif(single_feature_gen_content)
        # print(single_feature_dict)
        if single_feature_dict is not None:
            # print('line_number:', line_number)
            if line_number is not None:
                newline_pos, line_text = get_newline_pos(content=ori_file_content, line_number=line_number)
            else:
                newline_pos, line_text = 0, ori_file_content
            # print(newline_pos, line_text)
            answer_value = single_feature_dict.get("answer", "")  # Get the "answer" value in result_dict
            single_feature_dict["pos"] = find_keyword_pos(content=line_text, keyword=answer_value, offset=newline_pos)
            # single_feature_dict = {'phi': 'IDNUM', 'answer': '62', 'pos': [(59, 67), (68, 76), (59, 67), (68, 76)]}
            line_text = to_text_list(keyword_dict=single_feature_dict, file_name=file_name)
            # print('line_text:', line_text)
            text_list.extend(line_text) if line_text is not None else None
    # print("text_list:", text_list)
    if not text_list:
        return None
    text_list = remove_duplicates(text_list)
    output = '\n'.join(text_list)
    return output


if __name__ == '__main__':
    folder_path = '.\\test_dataset'
    gen_text_path = '.\\gen_text.json'
    results_path = '.\\test_results.txt'

    clear_file(results_path)

    # Read json to dictionary
    with open(gen_text_path, "r") as file:
        gen_text_dict = json.load(file)

    files = glob.glob(os.path.join(folder_path, '*.txt'))
    print("找到{}個檔案".format(len(files)))
    for file_path in files:
        results_list = []
        medical_report = read_file2text(path=file_path)
        fileName = os.path.splitext(os.path.basename(file_path))[0]
        # print('fileName:', fileName)

        for key, gen_text in gen_text_dict[fileName].items():
            index = int(key)
            # print('index: {}, text: {}'.format(index, gen_text))
            print(gen_text)
            result = gen_text_normalization(fileName, gen_text, medical_report, index + 1)
            if result is not None:
                results_list.append(result)
                print(result)

        # save to results_path
        with open(results_path, 'a', encoding='utf-8') as file:
            for result in results_list:
                file.write(str(result) + '\n')
        print("\033[93m------ Saved ------\033[0m")
    print("\033[92m------ Done ------\033[0m")
