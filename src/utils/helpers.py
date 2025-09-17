import os
import torch
import numpy as np


# def list_of_distances(X, Y):
#     return torch.sum(
#         (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1
#     )


# def make_one_hot(target, target_one_hot):
#     target = target.view(-1, 1)
#     target_one_hot.zero_()
#     target_one_hot.scatter_(dim=1, index=target, value=1.0)


# def makedir(path):
#     """
#     if path does not exist in the file system, create it
#     """
#     if not os.path.exists(path):
#         os.makedirs(path)


# def print_and_write(str, file):
#     print(str)
#     file.write(str + "\n")


# def find_high_activation_crop(activation_map, percentile=95):
#     threshold = np.percentile(activation_map, percentile)
#     mask = np.ones(activation_map.shape)
#     mask[activation_map < threshold] = 0
#     lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
#     for i in range(mask.shape[0]):
#         if np.amax(mask[i]) > 0.5:
#             lower_y = i
#             break
#     for i in reversed(range(mask.shape[0])):
#         if np.amax(mask[i]) > 0.5:
#             upper_y = i
#             break
#     for j in range(mask.shape[1]):
#         if np.amax(mask[:, j]) > 0.5:
#             lower_x = j
#             break
#     for j in reversed(range(mask.shape[1])):
#         if np.amax(mask[:, j]) > 0.5:
#             upper_x = j
#             break
#     return lower_y, upper_y + 1, lower_x, upper_x + 1


def load_ts_file(filepath):
    X = []
    y = []
    reading_data = False
    label_set = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # read class labels
            if line.startswith("@classLabel"):
                label_set = line.split()[2:]
                continue

            if line.startswith("@data"):
                reading_data = True
                continue

            if not reading_data or not line:
                continue

            # read data
            parts = line.split(":")
            dims = []
            for part in parts[:-1]:
                part = part.strip("()")
                dims.append([float(x) for x in part.split(",")])

            label = parts[-1].strip()

            if label not in label_set:
                continue

            # Prepare label index
            label_index = label_set.index(label)

            X.append(dims)
            y.append(label_index)

    return X, y
