import argparse
import yaml
import json


def strip_model_output_path(model_output_path):
    # TODO: strip output path etc.
    return None, None

def get_triplets(video_id, frame_id, num_frames, framerate):
    # Triplets are defined at 1 FPS, our video at given framerate
    start_frame = frame_id // framerate
    end_frame = (frame_id + num_frames) // framerate

    # TODO: build the path to the correct cholect50 annotations file
    annotations_path = f'./data/cholect50/labels/VID{video_id}.json'
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    # Index into the frames
    frame_annotations = annotations['annotations'][str(start_frame)]

    return None


def eval(model_output_path, num_frames, framerate):
    # TODO: load and initialize this properly
    load_splat_graph()

    # TODO: access/init the MLLM
    load_mllm()

    # TODO: strip the model output path to get the video id and frame id
    video_id, frame_id = strip_model_output_path(model_output_path)

    # TODO: index into cholect50 and retrieve the triplets
    triplets = get_triplets(video_id, frame_id, num_frames, framerate)



if __name__ == "__main__":
    # Parse arguments; receiving a yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Read the yaml file
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Get relevant configs
    model_output_path = config["model_output_path"]
    num_frames = config["num_frames"]
    framerate = config["framerate"]
    eval(model_output_path, num_frames, framerate)
