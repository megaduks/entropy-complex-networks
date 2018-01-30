import os
import json

this_dir, this_filename = os.path.split(__file__)
file = os.path.join(this_dir, "data", "networks.json")

with open(file) as data_file:
    __networks__ = json.load(data_file)

