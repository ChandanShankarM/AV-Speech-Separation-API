from video_features.utils import get_position, transformation_from_points
from video_features.model import LipNet

from AVDPRNN.data_loader import audio_reader
import os
import torch
from torch.nn.parallel import data_parallel
from AVDPRNN.model.avdprnn_model import Dual_RNN_model
from AVDPRNN.config.option import parse

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import numpy as np
import time
from video_features.model import LipNet
import torch.optim as optim
import re
import json
import tempfile
import shutil
import cv2
import face_alignment
import matplotlib.pyplot as plt
import uuid
from tqdm import tqdm, trange
from pathlib import Path
from joblib import Parallel, delayed
from time import perf_counter
import subprocess
import ffmpeg
import matplotlib.pyplot as plt

class inference():
  def __init__(self):
    self.model = LipNet()
    self.model = self.model.cuda()
    self.model.eval()
    self.net = nn.DataParallel(self.model).cuda()
    weights = 'video_features/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
    pretrained_dict = torch.load(weights)
    model_dict = self.model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    model_dict.update(pretrained_dict)
    self.model.load_state_dict(model_dict)

  def avss(self, file_path, coordinates):
    print("Converting Video to frames")
    p = tempfile.mkdtemp()
    stream = ffmpeg.input(file_path, nostdin=None)
    stream = ffmpeg.output(stream, f'{p}/%d.jpg', **{'qscale:v': 2, 'r':25}, loglevel="quiet")

    try:
        out, err = ffmpeg.run(stream)
    except:
        print("FFMPEG CRASHED")
        sys.exit(1)

    if err:
        print(err)
        sys.exit(1)

    files = os.listdir(p)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    #plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(p, '1.jpg')), cv2.COLOR_BGR2RGB))

    print("Face a")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=f'cuda:{0}')
    points = fa.get_landmarks_from_directory(p, show_progress_bar=True)
    points = sorted(points.items(), key=lambda x: int(Path(x[0]).stem))

    videos = [[], []]

    front256 = get_position(256)
    for filename, point_faces in tqdm(points):
        first_was_here, second_was_here = False, False
        for point in point_faces:        
            if first_was_here and second_was_here:
                print("Too many faces")
                break
            if coordinates[0][0] < max(point[:, 0]) < coordinates[0][1]:
                idx = 0
                first_was_here = True
            elif coordinates[1][0]< max(point[:, 0]) < coordinates[1][1]:
                idx = 1
                second_was_here = True
            else:
                continue
            if(point is not None):
                shape = np.array(point)
                shape = shape[17:]
                M = transformation_from_points(np.matrix(shape), np.matrix(front256))

                img = cv2.warpAffine(cv2.imread(filename), M[:2], (256, 256))
                (x, y) = front256[-20:].mean(0).astype(np.int32)
                w = 160//2
                img = img[y-w//2:y+w//2,x-w:x+w,...]
                img = cv2.resize(img, (128, 64))
                videos[idx].append(img)
            elif videos:
                videos[idx].append(videos[idx][-1])
            else:
                print("First frame is missing", file)
                
        if not first_was_here:
            print('first none', filename)
        if not second_was_here:
            print('second none', filename)
        
    if not videos[idx]:
        print("No faces in video")
        sys.exit(1)
            
    for idx in range(2):
        videos[idx] = np.stack(videos[idx], axis=0).astype(np.float32)
        videos[idx] = torch.FloatTensor(videos[idx].transpose(3, 0, 1, 2)) / 255.0

    face0_features = self.model(videos[0][None,...].cuda(), return_video_features=True).detach()
    face1_features = self.model(videos[1][None,...].cuda(), return_video_features=True).detach()

    reader = audio_reader.get_default_audio_reader()
    mix_audio = reader.load(file_path)
    mix_audio = reader.to_tensor(mix_audio)
    opt = parse('AVDPRNN/config/Dual_RNN/train.yaml')
    opt['Dual_Path_RNN']['upsample_size'] = mix_audio.shape[1] - 1
    net = Dual_RNN_model(**opt['Dual_Path_RNN'])

    dicts = torch.load('AVDPRNN/saved_model/trained_weights.pt', map_location='cpu')
    net.load_state_dict(dicts["model_state_dict"])
    net = net.to('cuda')
    net.eval()

    with torch.no_grad():
        batch = {"mix_audios": mix_audio.cuda(),
                "first_videos_features": face0_features.transpose(1, 0).cuda(),
                "second_videos_features": face1_features.transpose(1, 0).cuda(),
                "mix_noised_audios": mix_audio.cuda()}
        res = net(batch)

    import soundfile as sf
    for idx, s in enumerate(res):
        os.makedirs('output_audio/', exist_ok=True)
        filename="output_audio/"+f'test_{idx}.wav'
        sf.write(filename, s.detach().cpu().numpy().T, opt['datasets']['audio_setting']['sample_rate'])
