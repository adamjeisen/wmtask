from .analysis import compute_model_jacs, compute_model_rhs


class WMTaskEq:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def jac(self, hiddens, t=None, discrete=False):
        return compute_model_jacs(self.model, hiddens, self.params['dt'], self.params['tau'], discrete=discrete)

    def rhs(self, hiddens, t=None):
        return compute_model_rhs(self.model, hiddens, self.params['dt'], self.params['tau'])
