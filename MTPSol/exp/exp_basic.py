from models import MTPSol, MTPSol_mutil, MTPSol_seq, MTPSol_str


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MTPSol': MTPSol,
            # 'ProteinClassification': ProteinClassification,
            # 'ProteinLite': ProteinLite,
            'MTPSol_str' :MTPSol_str,
            'MTPSol_seq' : MTPSol_seq,
            'MTPSol_mutil': MTPSol_mutil
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def validate(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
