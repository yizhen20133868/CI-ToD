import models.base

class Model(models.base.Model):
    def get_pred(self, out):
        pred = []
        for ele in out:
            pred.append(ele)
        return pred