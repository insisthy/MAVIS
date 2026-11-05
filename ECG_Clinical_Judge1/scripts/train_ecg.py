"""Train the retained ResNet1D implementation on user-owned train/validation data."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ecg-index',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--waveform-root',type=Path,required=True)
    p.add_argument('--execute-training',action='store_true')
    a=p.parse_args(); from scripts._paths import private_output; a.output=private_output(a.output)
    if not a.execute_training:
        p.error('Training requires --execute-training')
    if a.output.exists():
        raise FileExistsError('Use a new output directory')
    import pandas as pd
    from src.training import ecg_resnet_training as core
    labels=pd.read_csv(a.labels)
    if set(labels['split'])!={'train','val'} or labels['admission_id'].duplicated().any():
        raise ValueError('Only unique, physically separated train/val records are allowed')
    core.INDEX_PATH=a.ecg_index.resolve();core.LABEL_PATH=a.labels.resolve()
    core.ROOT=a.waveform_root.resolve();core.EXPECTED_SPLIT_COUNTS=labels['split'].value_counts().to_dict()
    core.MODEL_DIR=a.output/'model';core.PREDICTION_DIR=a.output/'predictions';core.SELECTION_DIR=a.output/'selection';core.LOG_DIR=a.output/'logs'
    config=core.ScratchWaveformConfig(waveform_base_dir=str(a.waveform_root.resolve()))
    print(json.dumps(core.train(config,'no_weight'),indent=2))


if __name__=='__main__':
    main()
