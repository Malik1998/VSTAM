from data.dataset import VSTAMDataset
import argparse
from torch.utils.data import DataLoader
import numpy as np


parser = argparse.ArgumentParser(description='join jsons from detection')
parser.add_argument('--train_videos', default="dataset/train")
parser.add_argument('--train_masks', default="dataset/train_mask")
parser.add_argument('--exponential_frames_count', default=5, type=int)
parser.add_argument('--additional_by_score_count', default=3, type=int)
parser.add_argument('--previous_shot_count', default=2, type=int)
parser.add_argument('--batch_size', default=2, type=int)



if __name__ == '__main__':
    args = parser.parse_args()
    dataset = VSTAMDataset(args.train_videos, args.train_masks,
                           exponential_frames_count=args.exponential_frames_count,
                           additional_by_score_count=args.additional_by_score_count,
                           previous_shot_count=args.previous_shot_count
                           )

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    indexes_array = []
    idx_array = []
    for i, batch in enumerate(train_dataloader):
        img, label, idx, indexes = batch
        if i <= 5:
            print(idx, "id of the data object")
            print(indexes, "chosen frame numbers for each data object")
            print("--------iterate---------")
        indexes_array.append(indexes)
        idx_array.append(idx)


    print("CHANGING SCORES")
    for indexes, idx in zip(indexes_array, idx_array):
        for inds, id in zip(indexes, idx):
            id = id[0].item() # because this is array with one element on it
            dct_with_scores = {i.item() : np.random.rand() for i in inds}
            dataset.add_scores(id, dct_with_scores)

    print("CHANGED SCORES")
    for i, batch in enumerate(train_dataloader):
        img, label, idx, indexes = batch
        print(idx)
        print(indexes)
        print("iterate over")
        if i == 5:
            break