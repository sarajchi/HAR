import os
import pandas as pd


def check_image_csv_match(root_folder):
    mismatched_folders = []

    # 遍历根文件夹中的子文件夹
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        if os.path.isdir(subdir_path):
            # 找到子文件夹中的CSV文件
            csv_file = None
            images_count = 0

            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)

                if item.endswith('.csv'):
                    csv_file = item_path
                elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    images_count += 1

            # 如果找到了CSV文件，检查行数是否匹配
            if csv_file:
                csv_data = pd.read_csv(csv_file)
                csv_row_count = len(csv_data)

                if csv_row_count != images_count:
                    mismatched_folders.append(subdir)

    return mismatched_folders


# 使用示例
root_folder = './down'
mismatched_folders = check_image_csv_match(root_folder)

if mismatched_folders:
    print("以下文件夹中的图片数量与CSV行数不匹配:")
    for folder in mismatched_folders:
        print(folder)
else:
    print("所有文件夹中的图片数量与CSV行数匹配。")
