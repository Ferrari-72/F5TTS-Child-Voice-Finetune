import os
import argparse

def main(input_path, output_dir):
    print("Input path exists:", os.path.exists(input_path))
    
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析列名（第一行）
    header = lines[0].strip().split('\t')
    id_index = header.index('UTTRANS_ID')
    trans_index = header.index('TRANSCRIPTION')

    # 逐行处理数据
    for line in lines[1:]:
        if not line.strip():  # 跳过空行
            continue
        parts = line.strip().split('\t')
        if len(parts) < max(id_index, trans_index) + 1:
            continue  # 跳过列不全的行

        file_name = parts[id_index].split('.')[0]
        transcription = parts[trans_index]
        # print(file_name, transcription)

        output_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(transcription)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split transcription file into individual text files.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input .txt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output text files')

    args = parser.parse_args()
    main(args.input_path, args.output_dir)
