import os
from overcooked_ai_py.utils import load_dict_from_file

def read_layout_dict(layout_name, layout_dir):

    if os.path.abspath(__file__) != layout_dir:
        LAYOUTS_DIR = layout_dir
    else:
        LAYOUTS_DIR = os.path.dirname(os.path.abspath(__file__))

    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))
