import os
import pandas as pd
from torch.utils.data import Dataset
import cv2
import torch
import imageio
from torchvision import transforms as T


default_transforms = T.Compose([
    T.Resize((640, 480), T.InterpolationMode.BICUBIC),
    T.ToTensor(),
])

# we consider that folder contains videos
# we will iterate over all videos and all frames on it
class VSTAMDataset(Dataset):
    def __init__(self, video_dir,
                 exponential_frames_count=3,
                 additional_by_score_count=2,
                 previous_shot_count=1,
                 transform=default_transforms):
        self.exponential_frames_count = exponential_frames_count
        self.additional_by_score_count = additional_by_score_count
        self.previous_shot_count = previous_shot_count
        self.transform = transform

        self.all_videos = [os.path.join(video_dir, vid) for vid in os.listdir(video_dir)]
        # count of frame count on each video
        self.indexes = [imageio.mimread(v).shape[0] for v in self.all_videos]
        # stores {"video_idx":
        # {frame_idx:
        #   {"frame_idx - 1": score_1,
        #   "frame_idx - 2": score_2 },
        #
        #   .... }
        self.dict_additional_by_score = dict()


    def add_scores(self):
        pass

    def __len__(self):
        return sum(self.indexes)

    def get_idx(self, idx):
        for i in range(self.indexes):
            if idx < self.indexes[i]:
                return i, idx
            idx -= self.indexes

    # we will add 0 frame_idx, if there are no indexes
    def get_exponential_indexes(self, idx):
        frame_indexes = [idx] + [max(idx - 2 ** i, 0) for i in range(self.exponential_frames_count)]
        return frame_indexes

    # we will add last existing frame_idx, if there are no indexes
    def get_additional_indexes(self, idx, index_prev, dct_frames):
        previous_shot_idx = [index_prev[0] for _ in range(self.previous_shot_count)]
        if idx - 1 in dct_frames:
            previous_shot_idx = sorted(list(dct_frames[idx - 1].items()),
                                       key=lambda shot: shot[1],
                                       reverse=True)[:self.previous_shot_count]
        best_score_idx = [index_prev[0] for _ in range(self.additional_by_score_count)]

        dct_scores = dict()
        # we will change score according to most recent detections
        for i in range(idx - 1):
            for shot_idx, score in dct_frames[i].items():
                dct_scores[shot_idx] = score
        if len(dct_scores) > 0:
            # sort by score
            dct_scores = sorted(list(dct_scores.items()),
                                key=lambda shot: shot[1],
                                reverse=True)[:self.additional_by_score_count]
            best_score_idx = [shot[0] for shot in dct_scores]

        return best_score_idx + previous_shot_idx + index_prev

    def __getitem__(self, idx):

        video_idx, frame_idx = self.get_idx(idx)

        video = imageio.mimread(self.all_videos[video_idx])

        indexes = self.get_exponential_indexes(frame_idx)
        indexes = self.get_additional_indexes(frame_idx, indexes,
                                              self.dict_additional_by_score[video_idx])

        list_frames = []
        for ind in indexes:

            list_frames.append(video[ind] if self.transform is None else self.transform(video[ind]))
        list_frames = torch.cat(list_frames)
        return list_frames