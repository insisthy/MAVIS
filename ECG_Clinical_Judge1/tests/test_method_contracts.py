"""Data-free checks of original scientific rules; no LLM or model execution."""
import ast
import itertools
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def gate_functions():
    # Load the actual pure gate definitions without importing optional ML libraries.
    tree=ast.parse((ROOT/'src/agents/evidence_synthesis_agent.py').read_text())
    names={'V2_8_EVIDENCE_SUFFICIENCY','V2_8_MODALITY_ALIGNMENT','V2_8_OVERALL_CONFIDENCE','derive_overall_confidence_v2_8','deterministic_neutral_rag_gate_v2_8'}
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in names:
            nodes.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in names for x in n.targets):
            nodes.append(n)
    code=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));ns={}
    exec(compile(code,'original_gate','exec'),ns)
    return ns


class MethodContracts(unittest.TestCase):
    def test_gate_truth_table(self):
        gate=gate_functions()['deterministic_neutral_rag_gate_v2_8']
        for alignment,suff,confidence in itertools.product(['ALIGNED','DISAGREEMENT','UNCERTAIN'],['SUFFICIENT','INSUFFICIENT'],['HIGH','MODERATE','LOW']):
            expected=alignment=='DISAGREEMENT' or (suff=='INSUFFICIENT' and (confidence=='LOW' or alignment=='UNCERTAIN'))
            got=gate(evidence_sufficiency=suff,modality_alignment=alignment,overall_confidence=confidence)
            self.assertEqual(got['retrieval_decision']=='RETRIEVE',expected)

    def test_invalid_gate_fails_closed(self):
        gate=gate_functions()['deterministic_neutral_rag_gate_v2_8']
        for key in ['evidence_sufficiency','modality_alignment','overall_confidence']:
            args=dict(evidence_sufficiency='SUFFICIENT',modality_alignment='DISAGREEMENT',overall_confidence='HIGH');args[key]='INVALID'
            self.assertEqual(gate(**args)['retrieval_decision'],'SKIP')

    def test_confidence_is_stage_a_mapping(self):
        f=gate_functions()['derive_overall_confidence_v2_8']
        for alignment in ['ALIGNED','DISAGREEMENT','UNCERTAIN']:
            for status in ['CONSISTENT_CONFIDENT','COMPLEX_DISCORDANCE']:
                expected='HIGH' if alignment=='ALIGNED' and status=='CONSISTENT_CONFIDENT' else 'LOW' if alignment=='UNCERTAIN' else 'MODERATE'
                self.assertEqual(f({'global_consistency_status':status},evidence_sufficiency='SUFFICIENT',modality_alignment=alignment),expected)

    def test_label_boundary(self):
        from src.runtime.ecg_anchor_contract import assert_ecg_anchor_runtime_boundary
        for key in ['Target_CAG_Binary','group_label','split','FN']:
            with self.assertRaises(RuntimeError):
                assert_ecg_anchor_runtime_boundary({'nested':{key:1}})

    def test_exact_logit_projection(self):
        import math
        from src.runtime.final_single_controller import FinalSingleController
        policy=json.loads((ROOT/'configs/runtime.json').read_text())['correction_policy']
        checked=FinalSingleController.validate_correction_policy(policy)
        for delta in [policy['delta_up'],-policy['delta_down'],0.0]:
            p=0.4
            value=FinalSingleController._project_logit(p,delta,checked['probability_clip_epsilon'])
            self.assertAlmostEqual(value,1/(1+math.exp(-(math.log(p/(1-p))+delta))),places=14)

    def test_bad_numeric_policy_fails_closed(self):
        from src.runtime.final_single_controller import FinalSingleController
        p=json.loads((ROOT/'configs/runtime.json').read_text())['correction_policy'];p['delta_up']=float('nan')
        with self.assertRaises(RuntimeError):
            FinalSingleController.validate_correction_policy(p)

    def test_anonymous_output_directory(self):
        from scripts._paths import private_output
        with self.assertRaises(ValueError):
            private_output(ROOT/'src/generated_records')
        self.assertTrue(private_output(ROOT/'outputs/local').is_relative_to(ROOT/'outputs'))

    def test_configured_knowledge_cards_are_source_attributed(self):
        config=json.loads((ROOT/'configs/runtime.json').read_text())
        path=ROOT/config['knowledge_cards']
        cards=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertTrue(cards)
        self.assertEqual(len(cards),len({card['card_id'] for card in cards}))
        for card in cards:
            for field in ['card_title','evidence_sources','applicable_when','usage_boundary']:
                self.assertTrue(card.get(field))


if __name__=='__main__':
    unittest.main()
