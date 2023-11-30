import os 
from numpy.random import f, permutation, rand
from PIL import Image
import time
import torch
import random
import pickle5 as pickle
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import cv2


class VideoAnomalyDataset_C3D(Dataset):
    """Video Anomaly Dataset."""

    def __init__(self,
                 data_dir,
                 dataset='shanghaitech',
                 detect_dir=None,
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        # assert dataset in ['shanghaitech', 'ped2', 'avenue'], 'wrong type of dataset.'

        self.dataset = dataset
        self.data_dir = data_dir
        self.fliter_ratio = fliter_ratio
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        if ('train' in data_dir):
            self.test_stage = False
        elif ('test' in data_dir):
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'
        if not self.test_stage and self.dataset == 'shanghaitech':
            self.sample_step = 5
        elif self.dataset == 'ubnormal':
            self.sample_step = 5
        else:
            self.sample_step = 1

        if detect_dir != None:
            with open(detect_dir, 'rb') as f:
                self.detect = pickle.load(f)
        else:
            self.detect = None

        self.objects_list = []
        self.abnormal_list = []
        self.normal_list = []
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        contain = 0
        total_small_ = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(l)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                if self.detect is not None:
                    detect_result = self.detect[video_file][frame]
                    detect_result = detect_result[detect_result[:, 4] > self.fliter_ratio, :]
                    object_num = detect_result.shape[0]
                else:
                    object_num = 1.

                flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(
                    object_num, 0)
                is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
                is_contain = is_contain.any(-1)
                is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10
                width = detect_result[:, 2] - detect_result[:, 0]
                height = detect_result[:, 3] - detect_result[:, 1]
                # aspect_ratio = np.minimum(width / height, height / width)
                aspect_ratio = height / width
                for i in range(object_num):
                    if not is_contain[i]:
                        if not is_small[i]:
                            self.objects_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                      "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                            if self.dataset == 'ubnormal':
                                cls_label = 0
                                if video_file.split('_')[0] == 'abnormal':
                                    track_label_path = os.path.join('F:/UBnormal_frames/tracks_train',
                                                                    video_file + '.txt')
                                    track_label = np.loadtxt(track_label_path, dtype=str)
                                    idx_frame = []
                                    for tl in track_label:
                                        if int(tl.split(',')[1]) == frame:
                                            idx_frame.append(tl)
                                    for ol in idx_frame:
                                        x_min = int(ol.split(',')[2])
                                        y_min = int(ol.split(',')[3])
                                        x_max = int(ol.split(',')[4])
                                        y_max = int(ol.split(',')[5])

                                        xA = max(x_min, detect_result[i, :4][0])
                                        yA = max(y_min, detect_result[i, :4][1])
                                        xB = min(x_max, detect_result[i, :4][2])
                                        yB = min(y_max, detect_result[i, :4][3])

                                        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
                                        gtbox_Area = abs(max(x_max - x_min, 0) * max(y_max - y_min, 0))
                                        if interArea > 0:
                                            cls_label = 1
                                if cls_label == 0:
                                    self.normal_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                             "loc": detect_result[i, :4],
                                                             "aspect_ratio": aspect_ratio[i]})
                                else:
                                    self.abnormal_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                               "loc": detect_result[i, :4],
                                                               "aspect_ratio": aspect_ratio[i]})
                        else:
                            total_small_ += 1
                    else:
                        contain += 1

        print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s." \
              .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))

    def __len__(self):
        return len(self.objects_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0
        temproal_flag = True
        record = self.objects_list[idx]
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])

        cls_label = 0

        if not temproal_flag and not self.test_stage:
            if random.random() < 0.0001:
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)

        # if self.dataset == 'ped2':
        #     obj_ped2 = np.zeros((7, 64, 64))
        #     for t in range(self.frame_num):
        #         img_gray = cv2.cvtColor(np.transpose(obj[:, t, :, :]), cv2.COLOR_BGR2GRAY)
        #         obj_ped2[t] = img_gray
        #     obj_ped2 = np.expand_dims(obj_ped2, axis=0)
        #
        #     obj_js = self.jigsaw(obj_ped2, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        #     obj = torch.from_numpy(obj_ped2.copy())
        #     obj_js = torch.from_numpy(obj_js)
        # else:

        obj_js = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        obj = torch.from_numpy(obj.copy())
        obj_js = torch.from_numpy(obj_js)
        if self.dataset == 'ped2':
            a = []
            for i in range(3):
                a.append(obj)
            obj = torch.cat(a, dim=0)

        # NOT permute clips containing static contents
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if temproal_flag:
            obj_js = obj_js[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)
        obj_js = torch.clamp(obj_js,0.,1.)

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm,
               "trans_label": spatial_perm, "cls_label": cls_label,
               "loc": record["loc"], "aspect_ratio": record["aspect_ratio"], "temporal": temproal_flag,'obj_js':obj_js}
        return ret

    def get_object(self, video_name, frame, obj_id):
        video_dir = os.path.join('F:/', self.dataset, self.phase, 'patches', video_name)
        if self.dataset == 'ubnormal':
            video_dir = os.path.join('F:/', 'UBnormal_frames', self.phase, 'patches', video_name)
        obj = np.load(os.path.join(video_dir, str(frame) + '_' + str(obj_id) + '.npy'))  # (3, 9, 64, 64)
        if not self.test_stage:
            if random.random() < 0.5:
                obj = obj[:, :, :, ::-1]
        return obj

    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list

    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip

    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permuation=permuation, num=3,
                           dropout=dropout)
        return clip

class VideoAnomalyDataset_C3D_NM(Dataset):
    """Video Anomaly Dataset."""
    def __init__(self,
                 data_dir, 
                 dataset='shanghaitech',
                 detect_dir=None, 
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        # assert dataset in ['shanghaitech', 'ped2', 'avenue'], 'wrong type of dataset.'
        
        self.dataset = dataset
        self.data_dir = data_dir
        self.fliter_ratio = fliter_ratio
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        if('train' in data_dir):
            self.test_stage = False
        elif('test' in data_dir):
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'
        if not self.test_stage and self.dataset == 'shanghaitech':
            self.sample_step = 5
        elif self.dataset =='ubnormal':
            self.sample_step = 5
        else:
            self.sample_step = 1
        
        if detect_dir != None:
            with open(detect_dir, 'rb') as f:
                self.detect = pickle.load(f)
        else:
            self.detect = None

        self.objects_list = []
        self.abnormal_list = []
        self.normal_list = []
        self._load_data(file_list)
    
    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        contain = 0
        total_small_ = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(l)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                try:
                    if self.detect is not None:
                        try:
                            detect_result = self.detect[video_file][frame]
                        except Exception as e:
                            detect_result = self.detect[video_file][frame]
                        detect_result = detect_result[detect_result[:, 4] > self.fliter_ratio, :]
                        object_num = detect_result.shape[0]
                    else:
                        object_num = 1.

                    flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(object_num, 0)
                    is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
                    is_contain = is_contain.any(-1)
                    is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10
                    width = detect_result[:, 2] - detect_result[:, 0]
                    height = detect_result[:, 3] - detect_result[:, 1]
                    # aspect_ratio = np.minimum(width / height, height / width)
                    aspect_ratio = height / width
                    for i in range(object_num):
                        if not is_contain[i]:
                            if not is_small[i]:
                                self.objects_list.append({"video_name":video_file, "frame":frame, "object": i,
                                    "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                                if self.dataset == 'ubnormal':
                                    cls_label=0
                                    if video_file.split('_')[0] == 'abnormal':
                                        track_label_path = os.path.join('F:/UBnormal_frames/tracks_train',
                                                                        video_file + '.txt')
                                        track_label = np.loadtxt(track_label_path, dtype=str)
                                        idx_frame = []
                                        for tl in track_label:
                                            if int(tl.split(',')[1]) == frame:
                                                idx_frame.append(tl)
                                        for ol in idx_frame:
                                            x_min = int(ol.split(',')[2])
                                            y_min = int(ol.split(',')[3])
                                            x_max = int(ol.split(',')[4])
                                            y_max = int(ol.split(',')[5])

                                            xA = max(x_min, detect_result[i, :4][0])
                                            yA = max(y_min, detect_result[i, :4][1])
                                            xB = min(x_max, detect_result[i, :4][2])
                                            yB = min(y_max, detect_result[i, :4][3])

                                            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
                                            gtbox_Area = abs(max(x_max - x_min, 0) * max(y_max - y_min, 0))
                                            if interArea > 0:
                                                cls_label = 1
                                    if cls_label == 0:
                                        self.normal_list.append({"video_name":video_file, "frame":frame, "object": i,
                                        "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                                    else:
                                        self.abnormal_list.append({"video_name":video_file, "frame":frame, "object": i,
                                        "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                            else:
                                total_small_ += 1
                        else:
                            contain += 1
                except Exception:
                    continue

        print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s."\
            .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))

    def __len__(self):
        return len(self.normal_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0
        temproal_flag = True
        record = self.normal_list[idx]
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])

        cls_label = 0
        if not temproal_flag and not self.test_stage:
            if random.random() < 0.0001:
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)

        # if self.dataset == 'ped2':
        #     obj_ped2 = np.zeros((7, 64, 64))
        #     for t in range(self.frame_num):
        #         img_gray = cv2.cvtColor(np.transpose(obj[:, t, :, :]), cv2.COLOR_BGR2GRAY)
        #         obj_ped2[t] = img_gray
        #     obj_ped2 = np.expand_dims(obj_ped2, axis=0)
        #
        #     obj_js = self.jigsaw(obj_ped2, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        #     obj = torch.from_numpy(obj_ped2.copy())
        #     obj_js = torch.from_numpy(obj_js)
        # else:
        obj_js = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        obj = torch.from_numpy(obj.copy())
        obj_js = torch.from_numpy(obj_js)

        # NOT permute clips containing static contents
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if temproal_flag:
            obj_js = obj_js[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)
        obj_js = torch.clamp(obj_js, 0., 1.)

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm,
            "trans_label": spatial_perm,"cls_label":cls_label, "loc": record["loc"], "aspect_ratio": record["aspect_ratio"], "temporal": temproal_flag, "obj_js":obj_js}
        return  ret

    def get_object(self, video_name, frame, obj_id):
        video_dir = os.path.join('F:/', self.dataset, self.phase, 'patches', video_name)
        if self.dataset == 'ubnormal':
            video_dir = os.path.join('F:/', 'UBnormal_frames', self.phase, 'patches', video_name)
        obj = np.load(os.path.join(video_dir, str(frame) + '_' + str(obj_id) + '.npy'))  # (3, 9, 64, 64)
        if not self.test_stage:
            if random.random() < 0.5:
                obj = obj[:, :, :, ::-1]
        return obj

    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list

    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip

    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permuation=permuation, num=3,
                           dropout=dropout)
        return clip

class VideoAnomalyDataset_C3D_AM(Dataset):
    """Video Anomaly Dataset."""

    def __init__(self,
                 data_dir,
                 dataset='shanghaitech',
                 detect_dir=None,
                 fliter_ratio=0.9,
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        # assert dataset in ['shanghaitech', 'ped2', 'avenue'], 'wrong type of dataset.'

        self.dataset = dataset
        self.data_dir = data_dir
        self.fliter_ratio = fliter_ratio
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        if ('train' in data_dir):
            self.test_stage = False
        elif ('test' in data_dir):
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'
        if not self.test_stage and self.dataset == 'shanghaitech':
            self.sample_step = 5
        elif self.dataset == 'ubnormal':
            self.sample_step = 5
        else:
            self.sample_step = 1

        if detect_dir != None:
            with open(detect_dir, 'rb') as f:
                self.detect = pickle.load(f)
        else:
            self.detect = None

        self.objects_list = []
        self.abnormal_list = []
        self.normal_list = []
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        contain = 0
        total_small_ = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(l)
            total_frames += length
            # end_ind = length - start_ind

            # if self.dataset == 'ubnormal':
            #     cons_frames_idx = []
            #     for i in range(len(self.detect[video_file])):
            #         if len(self.detect[video_file][i]) > 0:
            #             cons_frames_idx.append(i)
            #     if len(cons_frames_idx) != length:
            #         cons_idx = []
            #         init_idx = 0
            #         for i in np.where(np.diff(np.array(cons_frames_idx)) != 1)[0]:
            #             if cons_frames_idx[i] - init_idx > self.sample_step * self.frame_num:
            #                 cons_idx.append([init_idx, cons_frames_idx[i]])
            #             if np.where(np.diff(np.array(cons_frames_idx)) != 1)[0][-1] == i:
            #                 if cons_frames_idx[-1]-cons_frames_idx[i] >self.sample_step * self.frame_num:
            #                     cons_idx.append([cons_frames_idx[i + 1],cons_frames_idx[-1]])
            #             init_idx = cons_frames_idx[i + 1]
            #         try:
            #             ind = random.choice(cons_idx)
            #         except Exception:
            #             if len(cons_frames_idx) != length and len(cons_idx) == 0:
            #                 ind = [cons_frames_idx[0], cons_frames_idx[-1]]
            #         if ind[0] == 0:
            #             ind[0] = start_ind
            #         if ind[-1] == length - 1:
            #             ind[-1] = length - start_ind
            #         start_ind = ind[0]
            #         end_ind = ind[-1]

            for frame in range(start_ind, length - start_ind, self.sample_step):
                try:
                    if self.detect is not None:
                        try:
                            detect_result = self.detect[video_file][frame]
                        except Exception as e:
                            detect_result = self.detect[video_file][frame]
                        detect_result = detect_result[detect_result[:, 4] > self.fliter_ratio, :]
                        object_num = detect_result.shape[0]
                    else:
                        object_num = 1.

                    flag = detect_result[:, None, :4].repeat(object_num, 1) - detect_result[None, :, :4].repeat(
                        object_num, 0)
                    is_contain = np.all(np.concatenate((flag[:, :, :2] > 0, flag[:, :, 2:] < 0), -1), -1)
                    is_contain = is_contain.any(-1)
                    is_small = (detect_result[:, 2:4] - detect_result[:, 0:2]).max(-1) < 10
                    width = detect_result[:, 2] - detect_result[:, 0]
                    height = detect_result[:, 3] - detect_result[:, 1]
                    # aspect_ratio = np.minimum(width / height, height / width)
                    aspect_ratio = height / width
                    for i in range(object_num):
                        if not is_contain[i]:
                            if not is_small[i]:
                                self.objects_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                          "loc": detect_result[i, :4],
                                                          "aspect_ratio": aspect_ratio[i]})
                                if self.dataset == 'ubnormal':
                                    cls_label = 0
                                    if video_file.split('_')[0] == 'abnormal':
                                        track_label_path = os.path.join('F:/UBnormal_frames/tracks_train',
                                                                        video_file + '.txt')
                                        track_label = np.loadtxt(track_label_path, dtype=str)
                                        idx_frame = []
                                        for tl in track_label:
                                            if int(tl.split(',')[1]) == frame:
                                                idx_frame.append(tl)
                                        for ol in idx_frame:
                                            x_min = int(ol.split(',')[2])
                                            y_min = int(ol.split(',')[3])
                                            x_max = int(ol.split(',')[4])
                                            y_max = int(ol.split(',')[5])

                                            xA = max(x_min, detect_result[i, :4][0])
                                            yA = max(y_min, detect_result[i, :4][1])
                                            xB = min(x_max, detect_result[i, :4][2])
                                            yB = min(y_max, detect_result[i, :4][3])

                                            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
                                            gtbox_Area = abs(max(x_max - x_min, 0) * max(y_max - y_min, 0))
                                            if interArea > 0:
                                                cls_label = 1
                                    if cls_label == 0:
                                        self.normal_list.append({"video_name": video_file, "frame": frame, "object": i,
                                                                 "loc": detect_result[i, :4],
                                                                 "aspect_ratio": aspect_ratio[i]})
                                    else:
                                        self.abnormal_list.append(
                                            {"video_name": video_file, "frame": frame, "object": i,
                                             "loc": detect_result[i, :4], "aspect_ratio": aspect_ratio[i]})
                            else:
                                total_small_ += 1
                        else:
                            contain += 1
                except Exception:
                    continue

        print("Load {} videos {} frames, {} objects, excluding {} inside objects and {} small objects in {} s." \
              .format(self.videos, total_frames, len(self.objects_list), contain, total_small_, time.time() - t0))

    def __len__(self):
        return len(self.abnormal_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx):
        temproal_flag = idx % 2 == 0
        record = self.abnormal_list[idx]
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)
        obj = self.get_object(record["video_name"], record["frame"], record["object"])

        cls_label = 1

        if not temproal_flag and not self.test_stage:
            if random.random() < 0.0001:
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)

        # if self.dataset == 'ped2':
        #     obj_ped2 = np.zeros((7, 64, 64))
        #     for t in range(self.frame_num):
        #         img_gray = cv2.cvtColor(np.transpose(obj[:, t, :, :]), cv2.COLOR_BGR2GRAY)
        #         obj_ped2[t] = img_gray
        #     obj_ped2 = np.expand_dims(obj_ped2, axis=0)
        #
        #     obj_js = self.jigsaw(obj_ped2, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        #     obj = torch.from_numpy(obj_ped2.copy())
        #     obj_js = torch.from_numpy(obj_js)
        # else:
        obj_js = self.jigsaw(obj, border=2, patch_size=20, permuation=spatial_perm, dropout=False)
        obj = torch.from_numpy(obj.copy())
        obj_js = torch.from_numpy(obj_js)

        # NOT permute clips containing static contents
        # if (obj[:, -1, :, :] - obj[:, 0, :, :]).abs().max() < self.static_threshold:
        #     perm = np.arange(self.frame_num)

        if temproal_flag:
            obj_js = obj_js[:, perm, :, :]
        obj = torch.clamp(obj, 0., 1.)
        obj_js = torch.clamp(obj_js, 0., 1.)

        ret = {"video": record["video_name"], "frame": record["frame"], "obj": obj, "label": perm,
               "trans_label": spatial_perm, "cls_label": cls_label,
               "loc": record["loc"], "aspect_ratio": record["aspect_ratio"], "temporal": temproal_flag, 'obj_js': obj_js}
        return ret
    

    def get_object(self, video_name, frame, obj_id):
        video_dir = os.path.join('F:/',self.dataset, self.phase,'patches', video_name)
        if self.dataset == 'ubnormal':
            video_dir = os.path.join('F:/','UBnormal_frames',self.phase,'patches',video_name)
        obj = np.load(os.path.join(video_dir, str(frame) + '_' + str(obj_id) + '.npy'))   # (3, 9, 64, 64)
        if not self.test_stage:
            if random.random() < 0.5:
                obj = obj[:, :, :, ::-1]
        return obj



    def split_image(self, clip, border=2, patch_size=20):
        """
        image: (C, T, H, W)
        """
        patch_list = []

        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])

        return patch_list


    def concat(self, patch_list, border=2, patch_size=20, permuation=np.arange(9), num=3, dropout=False):
        """
        batches: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permuation) - 1)
        for p_ind, i in enumerate(permuation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip


    def jigsaw(self, clip, border=2, patch_size=20, permuation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permuation=permuation, num=3, dropout=dropout)
        return clip
    
