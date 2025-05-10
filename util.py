import csv
import numpy as np
import os 
import json
import re
import pandas as pd
class Util:

    @staticmethod
    def write_file(output_file_path, result):
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=result[0].keys())
            writer.writeheader()
            for row in result:
                writer.writerow(row)


    @staticmethod
    def dump_jsonl(obj, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')

    
    @staticmethod
    def read_txt(file_path):
        assert str(file_path).endswith(".txt")
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        return data
    
    @staticmethod
    def read_json(file_path):
        assert str(file_path).endswith(".json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def append_json_to_file(data, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + '\n')  
            
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines
        
    @staticmethod
    def remove_inner_whitespace(text):
        pattern = r'(?<!^)\s+(?<!$)'
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text.replace(" ",'')
    
    @staticmethod
    def add_column_to_csv(file_path, array, new_column_name):
        if not os.path.exists(file_path):
            df = pd.DataFrame()
        else:
            if os.path.getsize(file_path) == 0:
                df = pd.DataFrame()
            else:
                df = pd.read_csv(file_path)
        new_column = pd.Series(array)
        if len(df) > 0 and len(new_column) != len(df):
            raise ValueError("erro")
        
        df[new_column_name] = new_column
        df.to_csv(file_path, index=False)
