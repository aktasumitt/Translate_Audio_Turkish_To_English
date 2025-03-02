from box import ConfigBox
import json
import os
import yaml
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys

def save_as_json(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "w",encoding="utf-8") as f:
            json.dump(data, f,ensure_ascii=False)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)

def load_json(path:Path):
    try:
        path=Path(path)
        with open(path, "r",encoding="utf-8") as f:
            loaded_data = json.load(f)
        return loaded_data    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_as_yaml(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, allow_unicode=True)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_yaml(path:Path):  
    try:     
        path=Path(path)
        with open(f"{path}", "r", encoding="utf-8") as file:
            loaded_dict = yaml.safe_load(file) 
        
        return ConfigBox(loaded_dict)
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)