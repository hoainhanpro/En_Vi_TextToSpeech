import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, visspeech


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    elif "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    elif "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    elif "ViSSpeech" in config["dataset"]:
        visspeech.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
