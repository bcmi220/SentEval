# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
SentenceTransformer models. See https://github.com/UKPLab/sentence-transformers.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
MODEL_PATH = '../pretrain/bert-base-nli-mean-tokens'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
sys.path.insert(1, '../sentence-transformers/')

import senteval

from sentence_transformers import SentenceTransformer

def prepare(params, samples):
    return


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.sentence_transformers.encode(sentences)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""
# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load Sentence Transformers model
    model = SentenceTransformer(MODEL_PATH)

    params_senteval['sentence_transformers'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    
    results = se.eval(transfer_tasks)

    print(results)
