"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import hashlib

import json

TAU_ECG = 0.3880248963832855

class ContractError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def validate_method(method):
    require(method['tau_ecg'] == TAU_ECG, 'METHOD_TAU_CHANGED')
    ecg=method['ecg']
    require((ecg['outer_folds'],ecg['inner_splits'],ecg['inner_selected_split'],ecg['seed'],ecg['refit_seed']) == (3,5,0,20260703,20260704), 'STRICT_OOF_PROTOCOL_CHANGED')
    require(method['expected_train_count'] > 0, 'EMPTY_TRAIN_COHORT')
