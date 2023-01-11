import torch
import os
import numpy as np
from MyDataLoader import loadDicomMultiFile
import MyUtils


MODEL_FOLDER_PATH = "output/"

landmarkNum = 17
use_gpu = 0
iteration = 3
cropSize = (32, 32, 32)
image_scale = (72, 96, 96)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstmModel = torch.load(MODEL_FOLDER_PATH + 'LSTM_model.pt', map_location=device)
corseNetModel = torch.load(MODEL_FOLDER_PATH + 'corseNet_model.pt', map_location=device)

img_path = os.path.join("processed_data/images/train/", "ceph1")
img = loadDicomMultiFile(img_path)

tensors = torch.tensor(np.array(img[0]))
tensors = tensors.to(device=device, dtype=torch.float32)
coarse_heatmap, coarse_features = corseNetModel(tensors)

gl, gh, gw = image_scale
global_coordinate = torch.ones(gl, gh, gw, 3).float()
for i in range(gl):
    global_coordinate[i, :, :, 0] = global_coordinate[i, :, :, 0] * i
for i in range(gh):
    global_coordinate[:, i, :, 1] = global_coordinate[:, i, :, 1] * i
for i in range(gw):
    global_coordinate[:, :, i, 2] = global_coordinate[:, :, i, 2] * i
global_coordinate = global_coordinate * torch.tensor([1 / (gl - 1), 1 / (gh - 1), 1 / (gw - 1)])

coarse_landmarks = MyUtils.get_coordinates_from_coarse_heatmaps(coarse_heatmap, global_coordinate).unsqueeze(0)
output = lstmModel(coarse_landmarks=coarse_landmarks, inputs_origin=tensors, phase='predict', coarse_feature=coarse_features)
# output['output'].size()

# output.keys()

heatmaps = output['output'].detach().cpu().numpy()
print(heatmaps)
# for heat in heatmaps:
#     cur_points = getPointsFromHeatmap(heat)
# return cur_points