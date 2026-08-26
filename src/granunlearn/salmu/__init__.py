"""SALMUBench hierarchical extension (Iteration 10).

Adapts the RELEASED SALMUBench artifacts without modifying them:
original metadata/datasets/models stay in the Hugging Face cache and are
referenced through ``salmu_original/`` manifests; every derived artifact
lives under ``salmu_hierarchical/`` (core hierarchies) or
``salmu_aux_redaction/`` (redaction-only identifiers).

Research goal: transform I_e <-> T(v_fine) into I_e <-> T(v_coarse) at
the controlled CLIP association level, mirroring the MLLMU experiment:
MF^SALMU / MG^SALMU / MN^SALMU reference states first, B0-B3 port only
after the separation gate.
"""
