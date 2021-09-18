# encoding=utf8
import json
import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class DatasetTool(object):
    @staticmethod
    def load_data(data_path):
        """
        Load the data from data path.
        :param data_path: the json file path
        :return: infos and consistency_tuples
        each <info> is the constructed format of dialogue.
        each <consistency_tuple> is a tuple of (qi,hi,kbi)
        """
        with open(data_path) as f:
            raw_data = json.load(f)
        domain = os.path.split(data_path)[-1].split("_")[0]
        data = []
        for dialogue_components_item in raw_data:
            constructed_info, last_response, consistency = DatasetTool.get_info(dialogue_components_item, domain)
            data_item = dict()
            data_item["constructed_info"] = constructed_info
            data_item["last_response"] = last_response
            data_item["consistency"] = consistency
            data.append(data_item)
        return data

    @staticmethod
    def get_info(dialogue_components_item, domain):
        """
        Transfer a dialogue item from the data
        :param dialogue_components_item: a dialogue(id, dialogue, kb, (qi,hi,kbi)) from data file (json item)
        :param domain: the domain of the data file
        :return: constructed_info: the constructed info which concat the info and format as
        the PhD. Qin mentioned.
                consistency: (qi,hi,kbi)
        """
        dialogue = dialogue_components_item["dialogue"]

        sentences = []
        history_sentences = []
        last_response = ''
        for speak_turn in dialogue:
            sentences.append(speak_turn["utterance"])
        if len(sentences) % 2 == 0:
            history_sentences.extend(sentences[:-1])
            last_response = sentences[-1]
        else:
            history_sentences.extend(sentences)

        knowledge_base = dialogue_components_item["scenario"]["kb"]['items']
        kb_expanded = DatasetTool.expand_kb(knowledge_base, domain)

        consistency = [float(x) for x in
                       [dialogue_components_item["scenario"]["qi"], dialogue_components_item["scenario"]["hi"],
                        dialogue_components_item["scenario"]["kbi"]]]

        constructed_info = DatasetTool.construct_info(kb_expanded, history_sentences)

        return constructed_info, last_response, consistency

    @staticmethod
    def expand_kb(knowledge_base, domain):
        """
        Expand the kb into (subject, relation, object) representation.
        :param knowledge_base: kb a list of dict.
        :param domain: the domain of the data
        :return: a list of list each item is a (subject, relation, object) representation.
        """
        expanded = []
        if domain == "navigate":
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['poi']
                for attribute_key in kb_row.keys():
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        elif domain == "calendar":
            if knowledge_base == None:
                return []
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['event']
                for attribute_key in kb_row.keys():
                    if kb_row[attribute_key] == "-":
                        continue
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        elif domain == "weather":
            for kb_row in knowledge_base:
                kb_row_list = []
                entity = kb_row['location']
                for attribute_key in kb_row.keys():
                    kb_row_list.append((entity, attribute_key, kb_row[attribute_key]))
                expanded.append(kb_row_list)
        else:
            print("Dataset is out of range(navigate, weather, calendar). Please recheck the path you have set.")
            assert False
        return expanded

    @staticmethod
    def construct_info(kb_expanded, history_sentences):
        """
        Concatenate the kb_expanded and history_sentences
        :param kb_expanded: the (subject, relation, object) representation expanded kb.
        :param history_sentences: history sentences.
        :return: the concatenated string.
        """
        construct_info = ''
        construct_info += ' [SOK] '
        for row in kb_expanded:
            construct_info += ' '.join([triple[1] + " " + triple[2] for triple in row])
            construct_info += ' ; '
        construct_info += ' [EOK] '

        for i, sentence in enumerate(history_sentences):
            if i % 2 == 0:
                construct_info += " [USR] " + sentence
            else:
                construct_info += " [SYS] " + sentence
        return construct_info

    @staticmethod
    def load_entity(args):
        entities = []
        for entity_path in args.dataset.entity.split(' '):
            with open(os.path.join(args.dir.dataset, entity_path)) as f:
                global_entity = json.load(f)
            entities.extend(DatasetTool.generate_entities(global_entity))
        return entities

    @staticmethod
    def generate_entities(global_entity):
        words = []
        for key in global_entity.keys():
            words.extend([str(x).lower().replace(" ", "_") for x in global_entity[key]])
            if '_' in key:
                words.append(key)
        return sorted(list(set(words)))

    @staticmethod
    def get(args, shuffle=True):
        """
        Get the train, dev, test data in a inner format of infos, last_responses, consistencys.
        ** infos means kb+history
        :param args:
        :return: train, dev, test data
        """
        train_paths = [os.path.join(args.dir.dataset, train_path) for train_path in args.dataset.train.split(' ')]
        dev_paths = [os.path.join(args.dir.dataset, dev_path) for dev_path in args.dataset.dev.split(' ')]
        test_paths = [os.path.join(args.dir.dataset, test_path) for test_path in args.dataset.test.split(' ')]
        train, dev, test = [], [], []
        [train.extend(DatasetTool.load_data(train_path)) for train_path in train_paths]
        [dev.extend(DatasetTool.load_data(dev_path)) for dev_path in dev_paths]
        [test.extend(DatasetTool.load_data(test_path)) for test_path in test_paths]
        if shuffle:
            np.random.shuffle(train)
            np.random.shuffle(dev)
        entities = DatasetTool.load_entity(args)
        return train, dev, test, entities

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
        summary["precision_qi"], summary["precision_hi"], summary["precision_kbi"] = precision_score(y_pred=pred_qi,
                                                                                                     y_true=gold_qi), precision_score(
            y_pred=pred_hi, y_true=gold_hi), precision_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["recall_qi"], summary["recall_hi"], summary["recall_kbi"] = recall_score(y_pred=pred_qi,
                                                                                         y_true=gold_qi), recall_score(
            y_pred=pred_hi, y_true=gold_hi), recall_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["f1_qi"], summary["f1_hi"], summary["f1_kbi"] = f1_score(y_pred=pred_qi, y_true=gold_qi), f1_score(
            y_pred=pred_hi, y_true=gold_hi), f1_score(y_pred=pred_kbi, y_true=gold_kbi)
        summary["overall_acc"] = sum(
            [1 for pred_i, gold_i in zip(pred, dataset) if pred_i == gold_i['consistency']]) / len(dataset)

        return summary

    @staticmethod
    def record(pred, dataset, set_name, args):
        pass
