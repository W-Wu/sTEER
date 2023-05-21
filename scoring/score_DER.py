import sys
from speechbrain.utils.DER import DER

ref_rttm = sys.argv[1]  #data/ref_rttms/fullref_iemo-test-5-VAD.rttm
sys_rttm = sys.argv[2]  #exp-eval/save/sys_rttms/IEMO_test-5/oracle_cos_SC/sys_output.rttm
skip_overlap=False
collar=0.25
[MS, FA, SER, DER_vals] = DER(
                ref_rttm,
                sys_rttm,
                skip_overlap,
                collar,
            )
print(DER_vals)

