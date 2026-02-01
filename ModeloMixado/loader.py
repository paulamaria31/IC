from numpy import savetxt, loadtxt
import random

import random
import numpy as np
from numpy import loadtxt

def load_data(folder_path, train_tasks, test_tasks, file_type, num_classes, verbose=0):
    train_content = []
    test_content = []

    all_subjects = list(range(1, 110))
    
    random.seed(1051)
    random.shuffle(all_subjects)

    training_subjects = all_subjects[:70]
    testing_subjects = all_subjects[70:]

    for i in training_subjects:
        for r in [1, 2]: # R01 (Olhos Abertos) e R02 (Olhos Fechados)
            path = f"{folder_path}S{i:03d}/S{i:03d}R{r:02d}.csv"
            train_content.append(loadtxt(path, delimiter=','))

    for i in testing_subjects:
        for r in [1, 2]:
            path = f"{folder_path}S{i:03d}/S{i:03d}R{r:02d}.csv"
            test_content.append(loadtxt(path, delimiter=','))

    return train_content, test_content
