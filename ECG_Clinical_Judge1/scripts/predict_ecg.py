"""Generate locked ECG scores from a trusted user-supplied checkpoint; no labels."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True,help='JSONL: case_id and absolute waveform_path (WFDB record without extension)')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from scripts._paths import private_output
    a.output=private_output(a.output)
    if not a.execute:
        p.error('Model inference requires --execute')
    if a.output.exists():
        raise FileExistsError('Use a new output file')
    from src.runtime.ecg_anchor_contract import assert_ecg_anchor_runtime_boundary
    rows=[json.loads(x) for x in a.index.read_text().splitlines() if x.strip()]
    ids=[row['case_id'] for row in rows]
    if not rows or len(ids)!=len(set(ids)):
        raise ValueError('Expected nonempty input with unique anonymous case_id values')
    for row in rows:
        assert_ecg_anchor_runtime_boundary(row)
    import numpy as np
    import torch
    import wfdb
    from src.models.resnet1d_small import ResNet1DSmall
    from src.data_processing.waveform_dataset import EXPECTED_LEADS,WaveformConfig,preprocess_waveform
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=ResNet1DSmall(in_channels=12,dropout=checkpoint.get('config',{}).get('dropout',0.3))
    model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    results=[]
    with torch.no_grad():
        for row in rows:
            record=wfdb.rdrecord(str(Path(row['waveform_path']).resolve()))
            if record.fs!=500 or record.sig_name!=EXPECTED_LEADS:
                raise ValueError('Expected 500-Hz 12-lead ECG in the native lead order')
            signal,_=preprocess_waveform(np.asarray(record.p_signal,dtype=np.float32).T,WaveformConfig())
            score=float(torch.sigmoid(model(torch.from_numpy(signal).unsqueeze(0))).reshape(-1)[0])
            results.append({'case_id':row['case_id'],'ecg_anchor_context':{'locked_ecg_score':score,'locked_ecg_threshold':0.3880248963832855}})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        for row in results:
            stream.write(json.dumps(row)+'\n')


if __name__=='__main__':
    main()
