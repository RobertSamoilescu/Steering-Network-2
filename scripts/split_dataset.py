#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_dir", type=str, help="split train/test directory")
args = parser.parse_args()

with open(os.path.join(args.split_dir, "train_scenes.txt"), "rt") as fin:
    train_scenes = fin.read()

with open(os.path.join(args.split_dir, "test_scenes.txt"), "rt") as fin:
    test_scenes = fin.read()
    
train_scenes = set(train_scenes.split("\n"))
test_scenes = set(test_scenes.split("\n"))

train_files = []
test_files = []

files = os.listdir("./dataset/img")

for file in files:
    scene, index, _ = file.split(".")
    index = int(index)

    if index == 0:
    	continue

    if scene in train_scenes:
        train_files.append(file[:-4])
    else:
        test_files.append(file[:-4])

# save as csv
train_csv = pd.DataFrame(train_files, columns=["name"])
test_csv = pd.DataFrame(test_files, columns=["name"])

train_csv.to_csv("./dataset/train.csv", index=False)
test_csv.to_csv("./dataset/test.csv", index=False)
