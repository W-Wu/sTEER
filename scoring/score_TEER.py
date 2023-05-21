import sys
sys.path.append('../')
from utils import *
from pyannote.metrics.identification import IdentificationErrorRate

class TEER_metric(IdentificationErrorRate):
    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super(TEER_metric, self).__init__(
            collar=collar, skip_overlap=skip_overlap, **kwargs)

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):
        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap,
            returns_uem=True)
        return super(TEER_metric, self).compute_components(reference, hypothesis, uem=uem,
                                collar=0.0, skip_overlap=False,
                                **kwargs)


skip_overlap=False
collar=0.25
print(collar)

metric_DER=TEER_metric(collar=collar*2,skip_overlap=skip_overlap)
# N.B.  
# NIST md-eval.pl tool define 'collar' as +/- collar seconds of a reference speaker segment boundary
# pyannote define 'collar' as collar seconds around reference segment boudaries. Needs to *2 to match the results using NIST md-eval.pl

modes = ['TEER','sTEER']
for mode in modes:
    if mode == 'TEER':
        print("test_id\tMS\tFA\tCOF\tTOTAL\tTEER")
    else:
        print("test_id\tMS\tFA\tCOF\tTOTAL\tsTEER")
    test_id= 5
    ref_rttm = f'../data/ref_emo_rttms/fullref_iemo-test-{test_id}-{mode}.rttm'
    sys_rttm = f'../exp-eval/ER_test_outcome-CAT-{test_id}-{mode}.rttm'

    component_dic={}
    DERs={}
    hyp = Rttm2dic(sys_rttm)
    ref = Rttm2dic(ref_rttm)

    for diag in hyp:
        component = metric_DER.compute_components(ref[diag], hyp[diag],uem=ref[diag].get_timeline().extent())
        # by setting uem=ref[diag], pyannote DER results are matched with NIST md-eval.pl toolkit
        for k,v in component.items():
            if k not in component_dic:
                component_dic[k]=v
            else:
                component_dic[k]+=v
        DERs[diag]=metric_DER.compute_metric(component)
    DERs['Total'] = metric_DER.compute_metric(component_dic)
    print(f"{test_id}\t{component_dic['missed detection']}\t{component_dic['false alarm']}\t{component_dic['confusion']}\t{component_dic['total']}\t{DERs['Total']}")
