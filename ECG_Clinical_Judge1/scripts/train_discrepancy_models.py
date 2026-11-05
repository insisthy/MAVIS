"""Train the two discrepancy heads with the original nested-CV selection code."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinical',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--ecg-oof',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-training',action='store_true')
    a=p.parse_args(); from scripts._paths import private_output; a.output=private_output(a.output)
    if not a.execute_training:
        p.error('Training requires --execute-training')
    import pandas as pd
    from src.training import discrepancy_training as core
    for path in [a.clinical,a.labels,a.ecg_oof]:
        frame=pd.read_csv(path)
        if 'split' not in frame or set(frame['split'].astype(str))!={'train'}:
            raise ValueError('Provide physically separate training-only files')
        if frame['admission_id'].duplicated().any():
            raise ValueError('Duplicate training keys')
    args=SimpleNamespace(agent2_input=str(a.clinical),label_table=str(a.labels),ecg_oof=str(a.ecg_oof),output_dir=str(a.output),ecg_threshold=0.3880248963832855,mode='formal_nested_cv',dataset_version='v2_9',grid_profile='formal_controlled',execute_training='true',test_allowed='false',bootstrap_n=0,outer_cv_splits=5,inner_cv_splits=3,elastic_net_max_iter=20000,head='all',disable_calibration=False,enable_sigmoid_calibration=True,disable_permutation_importance=True)
    core.run_training(args)


if __name__=='__main__':
    main()
