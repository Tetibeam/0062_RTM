# -*- coding: utf-8 -*-
import yaml
import os

def load_settings(path="setting.yaml"):
    """
    YAMLファイルから設定を読み込みます。

    Args:
        path (str): 設定ファイルのパス。

    Returns:
        dict: 読み込まれた設定。
    """
    with open(path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return settings

if __name__ == '__main__':
    # Test the function
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setting_path = os.path.join(base_dir, "..", "setting.yaml")
    settings = load_settings(setting_path)
    #print(settings)