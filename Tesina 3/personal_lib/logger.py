import numpy as np


class MLPLogger:
    """
    Classe per il Logger dell'MLP base
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.training_times = []
        self.train_scores = []
        self.test_scores = []
        self.loss_curves = []
        self.n_iterations = []
        self.sensitivities = [] # true positive rate
        self.specificities = [] # true negative rate
        self.aucs = [] # area under the curve

    def log_run(self, mlp, train_score, test_score, train_time, sens, spec, auc):
        self.training_times.append(train_time)
        self.train_scores.append(train_score)
        self.test_scores.append(test_score)
        self.loss_curves.append(mlp.loss_curve_)
        self.n_iterations.append(mlp.n_iter_)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.aucs.append(auc)

    def get_summary(self):
        return {
            'mean_train_acc': np.mean(self.train_scores),
            'std_train_acc': np.std(self.train_scores),
            'mean_test_acc': np.mean(self.test_scores),
            'std_test_acc': np.std(self.test_scores),
            'mean_time': np.mean(self.training_times),
            'std_time': np.std(self.training_times),
            'mean_sens': np.mean(self.sensitivities),
            'std_sens': np.std(self.sensitivities),
            'mean_spec': np.mean(self.specificities),
            'std_spec': np.std(self.specificities),
            'mean_auc': np.mean(self.aucs),
            'std_auc': np.std(self.aucs),
            'mean_iterations': np.mean(self.n_iterations)
        }


# Logger esteso per CV
class CVLogger:
    def __init__(self):
        self.train_scores = []
        self.training_times = []
        self.test_scores = []
        self.sensitivities = []
        self.specificities = []
        self.aucs = []
        self.n_iterations = []
        self.loss_curves = []

        self.y_true =[]
        self.y_pred =[]
        self.y_proba=[]

    def log_run(self, train_acc, test_acc, sens, spec, auc, train_time, n_iter, loss_curve, y_true, y_pred, y_proba):
        self.train_scores.append(train_acc)
        self.test_scores.append(test_acc)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.aucs.append(auc)
        self.training_times.append(train_time)
        self.n_iterations.append(n_iter)
        self.loss_curves.append(loss_curve)
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.y_proba.append(y_proba)

    def get_summary(self):
        return {
            'mean_train_acc': np.mean(self.train_scores),
            'std_train_acc': np.std(self.train_scores),
            'mean_test_acc': np.mean(self.test_scores),
            'std_test_acc': np.std(self.test_scores),
            'total_time' : np.sum(self.training_times),
            'mean_time': np.mean(self.training_times),
            'std_time': np.std(self.training_times),
            'mean_iterations': np.mean(self.n_iterations),
            'mean_sens': np.mean(self.sensitivities),
            'std_sens': np.std(self.sensitivities),
            'mean_spec': np.mean(self.specificities),
            'std_spec': np.std(self.specificities),
            'mean_auc': np.mean(self.aucs),
            'std_auc': np.std(self.aucs),
            'total_evaluations': len(self.test_scores)*5,
            'y_true' : self.y_true,
            'y_pred' : self.y_pred,
            'y_proba' : self.y_proba
        }