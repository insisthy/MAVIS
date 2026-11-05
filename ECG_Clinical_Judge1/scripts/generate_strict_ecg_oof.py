"""Generate strict training-only ECG OOF scores from user-owned waveforms."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='Training-only CSV; admission_id,waveform_id,Target_CAG_Binary,group_label,split,oof_fold_id')
    p.add_argument('--ecg-index',type=Path,required=True,help='CSV containing admission_id,waveform_id,hea_path,dat_path')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/strict_oof.json'))
    p.add_argument('--execute-training',action='store_true')
    a=p.parse_args(); from scripts._paths import private_output; a.output=private_output(a.output)
    if not a.execute_training:
        p.error('Training requires --execute-training')
    import pandas as pd
    from src.training.strict_oof.ecg_oof import run
    frame=pd.read_csv(a.manifest)
    m=json.loads(a.config.read_text());m.update(oof_manifest=str(a.manifest.resolve()),ecg_index=str(a.ecg_index.resolve()),expected_train_count=len(frame))
    rows=run(Path.cwd(),m,a.output.resolve())
    out=pd.DataFrame(rows).rename(columns={'ecg_oof_score':'ecg_score'})
    out['ecg_oof_model']='strict_resnet1d_oof';out['ecg_class_weight_strategy']='none'
    out.to_csv(a.output/'strict_ecg_oof.csv',index=False)


if __name__=='__main__':
    main()
