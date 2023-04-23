import os
import pandas as pd
from torch.utils.data import Dataset
import cv2
import torch
import imageio


# we consider that folder contains videos
# we will iterate over all videos and all frames on it
class VSTAMDataset(Dataset):
    def __init__(self, video_dir, mask_dir,
                 exponential_frames_count=3,
                 additional_by_score_count=2,
                 previous_shot_count=1):
        self.exponential_frames_count = exponential_frames_count
        self.additional_by_score_count = additional_by_score_count
        self.previous_shot_count = previous_shot_count

        self.all_videos = [os.path.join(video_dir, vid) for vid in os.listdir(video_dir)]
        self.all_masks = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)]
        # count of frame count on each video
        self.indexes = [len(imageio.mimread(v)) for v in self.all_videos]
        # stores {"video_idx":
        # {frame_idx:
        #   {"frame_idx - 1": score_1,
        #   "frame_idx - 2": score_2 },
        #
        #   .... }
        self.dict_additional_by_score = dict()


    def add_scores(self, idx, scores):
        video_idx, frame_idx = self.get_idx(idx)
        if video_idx not in self.dict_additional_by_score:
            self.dict_additional_by_score[video_idx] = dict()
        if frame_idx not in self.dict_additional_by_score[video_idx]:
            self.dict_additional_by_score[video_idx][frame_idx] = dict()
        self.dict_additional_by_score[video_idx][frame_idx].update(scores)

    def __len__(self):
        return sum(self.indexes)

    def get_idx(self, idx):
        for i in range(len(self.indexes)):
            if idx < self.indexes[i]:
                return i, idx
            idx -= self.indexes[i]

    # we will add 0 frame_idx, if there are no indexes
    def get_exponential_indexes(self, idx):
        frame_indexes =  [max(idx - 2 ** i, 0) for i in range(self.exponential_frames_count)][::-1] + [idx]
        return frame_indexes

    # we will add last existing frame_idx, if there are no indexes
    def get_additional_indexes(self, idx, index_prev, dct_frames):
        previous_shot_idx = [index_prev[0] for _ in range(self.previous_shot_count)]
        if None and idx - 1 in dct_frames:
            temp_dct = [(k, v) for k, v in dct_frames[idx - 1].items() if k not in index_prev]
            previous_shot_idx = sorted(temp_dct,
                                       key=lambda shot: shot[1],
                                       reverse=True)[:self.previous_shot_count]
        best_score_idx = [index_prev[0] for _ in range(self.additional_by_score_count)]

        dct_scores = dict()
        # we will change score according to most recent detections
        for i in range(idx - 1):
            if i in dct_frames:
                for shot_idx, score in dct_frames[i].items():
                    if shot_idx not in index_prev:
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
        mask = imageio.mimread(self.all_masks[video_idx])

        indexes = self.get_exponential_indexes(frame_idx)
        indexes = self.get_additional_indexes(frame_idx, indexes,
                                              self.dict_additional_by_score.get(video_idx, dict()))

        list_frames = []
        list_masks = []
        for ind in indexes:
            vid, mask_cur = video[ind], mask[ind]
            vid = torch.from_numpy(vid)
            mask_cur = torch.from_numpy(mask_cur)
            list_frames.append(vid)
            list_masks.append(mask_cur)
        list_frames = torch.cat(list_frames)
        list_masks = torch.cat(list_masks)
        return list_frames, list_masks, torch.LongTensor([idx]), torch.LongTensor(indexes)