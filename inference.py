# albumentations
import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

import copy
import glob
import sys
import torch

from feature_extractor import VideoMAEFeatureExtractor
from videomae_pooling import VideomaePooling

from fix_model_state_dict import fix_model_state_dict
#from get_transform_alb import get_transform
from read_yaml import read_yaml

LOCAL_TEST = False

if LOCAL_TEST:
    # local
    INPUT_PATH = "./input"
    OUTPUT_PATH = "./output"    
else:
    # docker
    INPUT_PATH = "/input"
    OUTPUT_PATH = "/output"
RESOURCE_PATH = "./resources"


def run(task, input_path, output_path):
    print("task:", task)
    assert(task == 'GRS' or task == 'OSATS')

    print("Setting up ...")
    
    # Read a config file
    cfg = read_yaml('videomae_global_pooling.yaml')

    print("Initializing feature extractor...")
    # setup feature extractor
    device = 'cuda'
    feature_extractor = VideoMAEFeatureExtractor(f"{RESOURCE_PATH}/{cfg.Model.videomae.pretrained}",
                                                 device,
                                                 target_fps=5)

    # pooing and MLP
    if task == 'GRS':
        num_outputs = cfg.Model.num_outputs.grs.total_num_outputs
    elif task == 'OSATS':
        num_outputs = cfg.Model.num_outputs.osats.total_num_outputs
    cfg.Model.mlp.params.hidden_channels = [256, num_outputs]

    # Define a model
    models = []
    if task == 'GRS':
        num_models = len(cfg.Model.mlp.pretrained_grs)
    elif task == 'OSATS':
        num_models = len(cfg.Model.mlp.pretrained_osats)
    # define a model
    base_model = VideomaePooling(cfg.Model.pooling,
                                 cfg.Model.mlp.params)
    for idx in range(num_models):
        model = copy.deepcopy(base_model)

        # Load pretrained model weights
        if task == 'GRS':
            print(f'  Loading: {cfg.Model.mlp.pretrained_grs[idx]}')
            checkpoint_path = f"{RESOURCE_PATH}/{cfg.Model.mlp.pretrained_grs[idx]}"            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        elif task == 'OSATS':
            print(f'  Loading: {cfg.Model.mlp.pretrained_osats[idx]}')
            checkpoint_path = f"{RESOURCE_PATH}/{cfg.Model.mlp.pretrained_osats[idx]}"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = fix_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        
        model.eval()
        model.to(device)
        
        models.append(model)

    # Get the input video files
    input_videos = glob.glob(f'{input_path}/*.mp4')

    # csv header
    if task == 'GRS':
        outputs = ['VIDEO,GRS']
    elif task == 'OSATS':
        outputs = ['VIDEO,OSATS_RESPECT,OSATS_MOTION,OSATS_INSTRUMENT,OSATS_SUTURE,OSATS_FLOW,OSATS_KNOWLEDGE,OSATS_PERFORMANCE,OSATS_FINAL_QUALITY'] 

    # for each video
    for input_video in input_videos:
        # feature extraction
        output = feature_extractor.extract_features_from_video(
            input_video, 
            clip_duration=cfg.Model.videomae.params.clip_duration, 
            overlap=cfg.Model.videomae.params.overlap
        )

        num_total_clips = len(output)
        features = []
        for clip_idx in range(num_total_clips):
            feature = output[clip_idx]['global_features']
            features.append(feature)
        features = torch.stack(features, dim=0)
        features = features.unsqueeze(0)
        features = features.to(device)
#        print("  features:", features.shape)  # (1, num_clips, feature_dim)

        # prediction
        print('Prediction')
        with torch.no_grad():
            ys = []
            for model in models:
                ys.append(model(features))
            ys = torch.cat(ys, dim=0)
            y = torch.mean(ys, dim=0)

        if task == "GRS":
            n = cfg.Model.num_outputs.grs.num_grs_classification
            logit = y[:n]              # [4]
            grs = torch.argmax(logit).item()

            s = f'{os.path.basename(input_video).split(".")[0]},{grs:d}'
            print(s)
            outputs.append(s)
        elif task == "OSATS":
            num = cfg.Model.num_outputs.osats.num_osats_classification
            yo = []
            s = 0
            for n in num:
                yo.append(y[s:s+n])
                s += n
            logit = torch.stack(yo, dim=0)  # (8,5)
            osats = torch.argmax(logit, dim=1)  # (8)

            s = f'{os.path.basename(input_video).split(".")[0]},'
            for score in osats:
                s += f'{score.item():d},'
            s = s[:-1]
            print(s)
            outputs.append(s)

    # write csv file
    if task == "GRS":
        output_file = f"{output_path}/predictions_GRS.csv"
    elif task == "OSATS":
        output_file = f"{output_path}/predictions_OSATS.csv"

    with open(output_file, 'wt') as f:
        for output in outputs:
            f.write(output + '\n')

    return 0

if __name__ == "__main__":
    task = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    raise SystemExit(run(task, input_path, output_path))

    """
    task = "GRS"
#    task = "OSATS"
    raise SystemExit(run(task, INPUT_PATH, OUTPUT_PATH))
    """

