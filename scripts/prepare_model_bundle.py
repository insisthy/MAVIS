"""Bind locally supplied, trusted Agent2 models; does not fit or predict."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['under-model','under-schema','over-model','over-schema','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/runtime.json'))
    a=p.parse_args(); from scripts._paths import private_output; a.output=private_output(a.output); config=json.loads(a.config.read_text())
    if a.output.exists():
        raise FileExistsError('Output must be a new directory')
    manifest={k:True for k in ['no_mock','no_fallback','scientific_agent2_models_runtime_locked','scientific_agent2_thresholds_locked','scientific_agent2_risk_bands_locked']}
    manifest.update(test_used=False,dataset_version='v2_9',threshold_lock_hash=sha(a.config))
    for name in ['under','over']:
        model=getattr(a,name+'_model').resolve();schema=getattr(a,name+'_schema').resolve()
        features=json.loads(schema.read_text())['features']
        from src.agents.clinical_assessment_agent import FORMAL_CLINICAL_FEATURES
        if not features or len(features)!=len(set(features)) or not set(features)<=set(FORMAL_CLINICAL_FEATURES)|{'oof_ecg_score'}:
            raise ValueError('Invalid feature schema')
        manifest.update({name+'_model_path':str(model),name+'_schema_path':str(schema),name+'_model_hash':sha(model),name+'_schema_hash':sha(schema),name+'_feature_setting':'F0_ecg_score_only' if features==['oof_ecg_score'] else 'F2_clinical_plus_ecg_score' if 'oof_ecg_score' in features else 'F1_clinical_only_discrepancy'})
    manifest.update(under_alert_threshold=config['under_lower'],under_high_risk_threshold=config['under_upper'],over_review_threshold=config['over_lower'],over_high_risk_threshold=config['over_upper'])
    a.output.mkdir(parents=True)
    (a.output/'runtime_lock_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Model bundle bound; no training or prediction executed.')


if __name__=='__main__':
    main()
