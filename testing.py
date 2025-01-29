import cv2
import argparse
from glob import glob
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import subprocess
import yaml

from event_based_downsampling import event_downsample
from utils import build_frames

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--data_folder', type=str, required=True)
parser.add_argument('-i', '--event_stream', type=str, required=True)

"""
--data_folder should be the desired event stream folder. e.g. corridor
In the EvDownsampling dataset, --event_stream is the name of desired event stream e.g. Arundel
"""

args = parser.parse_args(['--data_folder', 'corridor', '--event_stream', 'Arundel'])

path_to_data_folder = os.path.join(os.getcwd(), 'data', f'EvDownsampling_{args.data_folder}')

"""
Loading camera data numpy files
"""

np_files = glob(f'{path_to_data_folder}/numpy/EvDownsampling_{args.event_stream}_*.npy')

def subprocess_run():
    subprocess.run(['python', 'dualCam_dvRead.py', 
                    '--data_folder', f'{args.data_folder}', 
                    '--input', f'{args.event_stream}'])

if not np_files:
    # Run subprocess_run() if camera data numpy files do not exist in directory
    subprocess_run()
    np_files = glob(f'{path_to_data_folder}/numpy/EvDownsampling_{args.event_stream}_*.npy')

# Load record arrays of DVS camera data
Davis, Dvxplorer = [np.load(file) for file in sorted(np_files)]

# Load params
with open('EvDownsampling_config.yaml') as stream:
    params = yaml.safe_load(stream)

"""
Downsample -- example downsampling of Dvxplorer events
"""

# Downsample events with parameters mentioned in the paper
downsampled_events = event_downsample(events=Dvxplorer, 
                                      sensor_size=tuple(params["downsampling"]["sensor_size"]), 
                                      target_size=tuple(params["downsampling"]["target_size"]), 
                                      dt=params["downsampling"]["dt"], 
                                      tau_theta=params["downsampling"]["tau_theta"], 
                                      tau_accumulator=params["downsampling"]["tau_accumulator"], 
                                      beta=params["downsampling"]["beta"])

original_frames = build_frames(events=Dvxplorer,
                                 camera_name='Original', 
                                 time_resolution=params["visualisation"]["time_resolution"],
                                 output_resolution=tuple(params["downsampling"]["sensor_size"][:2]))

downsampled_frames = build_frames(events=downsampled_events, 
                                  camera_name=params["visualisation"]["camera_name"], 
                                  time_resolution=params["visualisation"]["time_resolution"],
                                  output_resolution=tuple(params["downsampling"]["target_size"]))


def write_to_video_side_by_side(original_frames, downsampled_frames, time_resolution, window_size):
    # Write to video
    out_size = (window_size[0]*2, window_size[1])
    vid = cv2.VideoWriter(f'output_side_by_side.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, out_size)

    print(original_frames.shape)
    print(downsampled_frames.shape)

    for f in range(downsampled_frames.shape[0]):

        original_image = original_frames[f,...]
        original_image[...,[1, 2]] = original_image[...,[2, 1]]
        
        downsampled_image = downsampled_frames[f,...]
        downsampled_image[...,[1, 2]] = downsampled_image[...,[2, 1]]
        
        resized_original = cv2.resize(original_image, window_size)
        resized_downsampled = cv2.resize(downsampled_image, window_size)
        
        side_by_side = np.hstack((resized_original, resized_downsampled))
        
        cv2.putText(side_by_side, text=f'Original, t={f*time_resolution} ms', 
                    org=(0,int(30/window_size[1]*window_size[1])), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=500/window_size[1], 
                    color=(255, 255, 255))
        
        cv2.putText(side_by_side, text=f'Downsampled, t={f*time_resolution} ms', 
                    org=(int(window_size[1]),int(30/window_size[1]*window_size[1])), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=500/window_size[1], 
                    color=(255, 255, 255))
        
        # Write to video
        vid.write(side_by_side.astype(np.uint8))

    vid.release()


def write_to_video(camera_name, frames, time_resolution, window_size):
    # Write to video
    vid = cv2.VideoWriter(f'output_{camera_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, window_size)

    for f in range(frames.shape[0]):

        image = frames[f,...]
        image[...,[1, 2]] = image[...,[2, 1]]
        
        resized = cv2.resize(image, window_size)
        
        cv2.putText(resized, text=f'{camera_name}, t={f*time_resolution} ms', 
                    org=(0,int(30/window_size[1]*window_size[1])), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=500/window_size[1], 
                    color=(255, 255, 255))
        
        # Write to video
        vid.write(resized.astype(np.uint8))

    vid.release()


def visualisation(camera_name, frames, time_resolution, window_size):
    for f in range(frames.shape[0]):
        
        cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_name, window_size, interpolation=cv2.INTER_AREA)
        
        image = frames[f,...]
        image[...,[1, 2]] = image[...,[2, 1]]
        
        resized = cv2.resize(image, window_size)
        
        cv2.putText(resized, text=f'{camera_name}, t={f*time_resolution} ms', 
                    org=(0,int(30/window_size[1]*window_size[1])), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=500/window_size[1], 
                    color=(255, 255, 255))
        
        cv2.imshow(camera_name, resized.astype(np.uint8))
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()    


write_to_video_side_by_side(original_frames=original_frames, 
                            downsampled_frames=downsampled_frames, 
                            time_resolution=params["visualisation"]["time_resolution"], 
                            window_size=tuple(params["visualisation"]["window_size"]))


#write_to_video(camera_name=params["visualisation"]["camera_name"], 
#              frames=downsampled_frames, 
#              time_resolution=params["visualisation"]["time_resolution"], 
#              window_size=tuple(params["visualisation"]["window_size"]))
