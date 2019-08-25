#!/usr/bin/env python

import numpy as np
import torch
import pickle
from models import GPCNNLSTM

def move_to_cuda(input_data): 
    input_data_ = []
    for d in input_data:
        input_data_.append(d.cuda())
    return input_data_

    def postprocess(prediction):
        sep_flag = 0
        pr = []
        for output in prediction:
            if sep_flag == 1:
                pr.append(1)
                continue
            if output[1] > 0.4:
                sep_flag = 1
                pr.append(1)
            else:
                pr.append(0)
        return pr

def get_sepsis_score(data, model):
    varofint = ['HR','O2Sat','Temp','SBP','MAP','DBP', 'Resp']
    d = pd.DataFrame(data=data_mat[:,0:7], columns=varofint)
    for i, _ in enumerate(varofint):
        if d[col].count() == 0:
            d[col] = d[col].fillna(0)
        elif d[col].count() == 1:
            d[col] = d[col].fillna(d[col].mean())
        elif d[col].count() <= 3 :
            d[col] = d[col].interpolate(method='linear').interpolate(method='nearest')

        if d[col].count() > 3:
            d[col] = d[col].interpolate(method='spline', order=3, axis=0).interpolate(method='nearest')
            d[col] = np.log(1+d[col])
            d[col] = (d[col] - d[col].mean())/d[col].std(ddof=0)
            d[col] = d[col].fillna(0.0)

    vital_features = torch.FloatTensor(d.values)
    input_data = [vital_features.unsqueeze(0), [], []]

    input_data_ = move_to_cuda(input_data)
    labels_ = labels.cuda()
    output = model.predict(input_data_)   
    output = output.tolist()   
    score = [o[0] for o in output][-1]
    label = postprocess(output)[-1]

    return score, label

def load_sepsis_model():
    model =  GPCNNLSTM((0,7))
    model.load_state_dict(torch.load('./model.out'))
    return model
