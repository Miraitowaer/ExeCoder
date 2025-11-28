import json
import os
import argparse
import glob
from tqdm import tqdm

def main(args):
    # 构造文件搜索路径
    search_path = os.path.join(args.input_dir, args.pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"Error: No files found matching pattern: {search_path}")
        return

    print(f"Found {len(files)} files to merge from {args.input_dir}...")
    
    all_data = []
    
    # 遍历并读取所有文件
    for file_path in tqdm(files, desc="Merging files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: File content in {file_path} is not a list, skipping.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}, skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Merge complete! Total samples: {len(all_data)}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 保存合并后的文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved merged dataset to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Directory where batch files are located")
    parser.add_argument('--output_file', type=str, required=True, help="Final merged output JSON file path")
    parser.add_argument('--pattern', type=str, default="dpo_batch_*_pairs_debug.json", help="Filename pattern to match (default: dpo_batch_*_pairs_debug.json)")
    args = parser.parse_args()
    main(args)