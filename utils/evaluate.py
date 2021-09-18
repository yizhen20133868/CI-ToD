# encoding=utf8
import os

from sklearn.metrics import precision_score, recall_score, f1_score

class EvaluateTool(object):

    @staticmethod
    def evaluate(pred, dataset, args):
        pred_qi = [pred_i[0] for pred_i in pred]
        pred_hi = [pred_i[1] for pred_i in pred]
        pred_kbi = [pred_i[2] for pred_i in pred]

        gold_qi = [gold_i['consistency'][0] for gold_i in dataset]
        gold_hi = [gold_i['consistency'][1] for gold_i in dataset]
        gold_kbi = [gold_i['consistency'][2] for gold_i in dataset]
        summary = {}

        if not os.path.exists(args.dir.output):
            os.makedirs(args.dir.output)
        summary["precision_qi"], summary["precision_hi"], summary["precision_kbi"] = precision_score(y_pred=pred_qi, y_true=gold_qi), precision_score(y_pred=pred_hi, y_true=gold_hi), precision_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["recall_qi"], summary["recall_hi"], summary["recall_kbi"] = recall_score(y_pred=pred_qi,y_true=gold_qi), recall_score(y_pred=pred_hi, y_true=gold_hi), recall_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["f1_qi"], summary["f1_hi"], summary["f1_kbi"] = f1_score(y_pred=pred_qi, y_true=gold_qi), f1_score(y_pred=pred_hi, y_true=gold_hi), f1_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["overall_acc"] = sum([1 for pred_i, gold_i in zip(pred, dataset) if pred_i == gold_i['consistency']]) / len(dataset)

        return summary
