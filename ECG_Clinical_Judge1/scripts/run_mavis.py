"""Run MAVIS on a user-owned, label-free JSONL input and local model bundle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def checked_inputs(path, threshold):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty input')
    from src.runtime.ecg_anchor_contract import assert_ecg_anchor_runtime_boundary
    seen=set()
    for row in rows:
        assert_ecg_anchor_runtime_boundary(row)
        case=row.get('case_id')
        if not isinstance(case,str) or not case or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case) or case in seen:
            raise ValueError('Use unique anonymous case_id values with letters/digits/_/- only')
        seen.add(case)
        anchor=row.get('ecg_anchor_context',{})
        score=float(anchor['locked_ecg_score'])
        if not math.isfinite(score) or not 0<=score<=1 or float(anchor['locked_ecg_threshold'])!=threshold:
            raise ValueError('Invalid ECG score or unlocked threshold')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--model-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/runtime.json'))
    p.add_argument('--execute',action='store_true',help='Explicitly enable model inference and LLM calls')
    a=p.parse_args(); from scripts._paths import private_output; a.output=private_output(a.output);cfg=json.loads(a.config.read_text());rows=checked_inputs(a.input,cfg['ecg_threshold'])
    manifest=a.model_bundle.resolve()/'runtime_lock_manifest.json'
    if not manifest.exists():
        raise FileNotFoundError('Run prepare_model_bundle first')
    from src.models.clinical_discrepancy_model import _verify_runtime_lock_manifest
    model_cfg={'runtime_lock_manifest':str(manifest),'threshold_lock_config':str(a.config.resolve()),'mock_allowed':False,'fallback_allowed':False,'scientific_agent2_models_runtime_locked':True,'scientific_agent2_thresholds_locked':True,'scientific_agent2_risk_bands_locked':True,'stage_a_semantic_commit_enabled':True,'continuous_numeric_directional_semantics_committed':False}
    _verify_runtime_lock_manifest(model_cfg)
    if not a.execute:
        print('Input/bundle checks passed. No model inference or LLM calls. Add --execute to run.')
        return
    if a.output.exists():
        raise FileExistsError('Use a new output directory; automatic resubmission is disabled')
    from src.llm.llm_client import load_llm_config
    from src.runtime.wire import StructuredClient
    from src.runtime.orchestration import RawResponsePersistingClient,build_agents,CaseTransaction,process_case,atomic_write_json,stable_hash
    llm=load_llm_config()
    if not llm.api_key or not llm.api_base or not llm.model:
        raise ValueError('LLM_API_KEY, LLM_API_BASE and LLM_MODEL are required')
    llm.connect_timeout_seconds=30;llm.read_timeout_seconds=240;llm.temperature=0.0
    a.output.mkdir(parents=True)
    client=RawResponsePersistingClient(StructuredClient(llm),a.output/'private/raw_responses',runtime_version='v2.9',max_technical_retries_per_call=1,retry_backoff_seconds=[10,30],agent1_3_streaming_enabled=True,agent1_3_max_stream_duration_seconds=240,agent1_3_thinking_mode='disabled',agent4_streaming_enabled=True,agent4_max_output_tokens=1536,agent4_max_total_attempts=2,agent4_thinking_mode='disabled')
    settings=SimpleNamespace(llm_max_retries=0,llm_retry_backoff_seconds=0,llm_retry_backoff_multiplier=1,agent4_max_generation_attempts=2,agent4_max_empty_content_regeneration_attempts=1)
    def sink(trace):
        # Raw/committed traces are private local outputs, excluded by .gitignore.
        atomic_write_json(a.output/'private/traces'/f"{trace['case_id']}_{trace['agent_id']}_{stable_hash(trace)}.json",trace)
    agents=build_agents(client,settings,sink)
    from src.agents.prompts import V2_6_ACTIVE_RUNTIME_PROMPTS
    from src.agents.schemas import V2_6_AGENT_SCHEMAS
    code_hash={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(Path('src').rglob('*.py'))}
    fingerprint={'code':code_hash,'config':cfg,'model_bundle':json.loads(manifest.read_text()),'model':llm.model,'api_base':llm.api_base}
    for row in rows:
        row={**row,'dataset_version':'v2_9','canonical_relation_v3_1_enabled':True,'runtime_graded_directional_mode_enabled':True,'runtime_controller_direction_guard_enabled':True,'runtime_correction_policy_locked':True,'runtime_policy_variant':'final_single_controller_validation_locked_logit_v1','agent2_discrepancy_model_config':model_cfg}
        tx=CaseTransaction(a.output,row['case_id'],stable_hash(row),stable_hash(V2_6_ACTIVE_RUNTIME_PROMPTS),stable_hash(V2_6_AGENT_SCHEMAS),agents[0].model_inference_config_hash(),agents[0].transport_execution_config_hash(),'mavis_public_v1','sha256',fingerprint)
        result=process_case(row,agents=agents,transaction=tx,correction_policy=cfg['correction_policy'],knowledge_cards_file=Path(cfg['knowledge_cards']).resolve())
        with (a.output/'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__':
    main()
