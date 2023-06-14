#  Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from os.path import basename as bn
from pathlib import Path

import numpy as np

import os
import sys
if 'MAPILLARY_ROOT' in os.environ:
    sys.path.append(os.environ['MAPILLARY_ROOT'])
else:
    # get current file's path, go up one level and add libs/mapillary_sls to path
    file_path = os.path.dirname(os.path.realpath(__file__))
    project_path = os.path.abspath(os.path.join(file_path, os.pardir))
    mapillary_path = os.path.join(project_path, 'libs', 'mapillary_sls')

    sys.path.append(mapillary_path)
    

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import eval


def validate(prediction, msls_root, result_file, ks=[1, 5, 10, 20]):

    # select for which ks to evaluate

    dataset = MSLS(msls_root, cities="", mode='val', posDistThr=25)

    # get query and positive image keys
    database_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages]
    positive_keys = [[','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages[pos]] for pos in dataset.pIdx]
    query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages[dataset.qIdx]]
    all_query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages]

    # load prediction rankings
    predictions = np.loadtxt(prediction, ndmin=2, dtype=str)

    # Ensure that there is a prediction for each query image
    for k in query_keys:
        assert k in predictions[:, 0], "You didn't provide any predictions for image {}".format(k)

    # Ensure that all predictions are in database
    for i, k in enumerate(predictions[:, 1:]):
        missing_elem_in_database = np.in1d(k, database_keys, invert = True)
        if missing_elem_in_database.all():

            print("Some of your predictions are not in the database for the selected task {}".format(k[missing_elem_in_database]))
            print("This is probably because they are panorama images. They will be ignored in evaluation")

            # move missing elements to the last positions of prediction
            predictions[i, 1:] = np.concatenate([k[np.invert(missing_elem_in_database)], k[missing_elem_in_database]])

    # Ensure that all predictions are unique
    for k in range(len(query_keys)):
        assert len(predictions[k, 1:]) == len(np.unique(predictions[k, 1:])), "You have duplicate predictions for image {} at line {}".format(query_keys[k], k)

    # Ensure that all query images are unique
    assert len(predictions[:,0]) == len(np.unique(predictions[:,0])), "You have duplicate query images"

    predictions = np.array([l for l in predictions if l[0] in query_keys])

    # evaluate ranks
    metrics = eval(query_keys, positive_keys, predictions, ks=ks)

    f = open(result_file, 'w') if result_file else None
    # save metrics
    for metric in ['recall', 'map']:
        for k in ks:
            line =  '{}_{}@{}: {:.3f}'.format("all",
                                              metric,
                                              k,
                                              metrics['{}@{}'.format(metric, k)])
            print(line)
            if f:
                f.write(line + '\n')
    if f:
        f.close()
    return metrics



