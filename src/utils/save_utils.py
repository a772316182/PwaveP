import os


def pre_check_path(path: str) -> str:
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return path

import csv

def export_to_csv(data, filename):
    """
    将由字典组成的数组导出为 CSV 文件，并自动生成表头。

    参数:
    data (list of dict): 要导出的数据，每个元素是一个字典。
    filename (str): 导出的 CSV 文件名。
    """
    if not data:
        print("数据为空，未生成CSV文件。")
        return

    # 获取所有可能的字段名作为表头（取第一个字典的键）
    headers = data[0].keys()

    # 确保所有字典都包含相同的键
    for item in data:
        if item.keys() != headers:
            raise ValueError("数据中存在不一致的字段，无法导出为CSV。")

    # 写入CSV文件
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"数据已成功导出至 {filename}")