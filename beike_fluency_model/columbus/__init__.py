#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))

import getopt
import pandas as pd
from util.wrapFeature import *
from init.init import readConfig
import warnings
import pickle
import numpy as np
import json
# from config import Config

config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "class_sample.ini")
g_config = readConfig(config_path)

cols = ['AvgCharPerSent', 'AvgSpeed', 'CharNumVar', 'CharNum_25Per',
        'CharNum_50Per', 'CharNum_75Per', 'CharNum_Avg', 'CharNum_Max',
        'CharNum_Min', 'CharNum_Std', 'CharNum_Var', 'Duplicate1Abs',
        'Duplicate1Percent', 'Duplicate2Abs', 'Duplicate2Percent',
        'EnWordPercent', 'FileLen', 'LongBlankNum', 'LongChatNum',
        'LongSentPercent', 'SentLen_25Per', 'SentLen_50Per', 'SentLen_75Per',
        'SentLen_Avg', 'SentLen_Max', 'SentLen_Min', 'SentLen_Std',
        'SentLen_Var', 'SentSpeed_25Per', 'SentSpeed_50Per', 'SentSpeed_75Per',
        'SentSpeed_Avg', 'SentSpeed_Max', 'SentSpeed_Min', 'SentSpeed_Std',
        'SentSpeed_Var', 'SpeedByFile', 'SpeedByVoice', 'SpeedVar',
        'TotalAdjNum', 'TotalAdvNum', 'TotalCharNum', 'TotalNounNum',
        'TotalPauseWordNum', 'TotalQuestionSentNum', 'TotalSentNum',
        'TotalVerbNum', 'TotalXNum', 'VoiceLen', 'VoiceOverFilePercent']


def single_file_processor(configObj, wavPath, asrDf):
    file_name = wavPath.split('/')[-1]
    basic_feature_df, _cache = getBasicFeatures(configObj, wavPath, asrDf)
    # this line we will try to arrange the order of all
    # if len(basic_feature_df) == 0:
    #     return pd.Series([file_name], index=['FileName'])
    trans_feature_df = getTransFeatures(configObj, basic_feature_df, _cache)
    cross_feature_df = getCrossetCrossFeatures(configObj, basic_feature_df, _cache)
    file_name_df = pd.Series([file_name], index=['FileName'])
    # allFeatures = pd.concat([basic_feature_df, trans_feature_df, cross_feature_df], axis=1)
    allFeatures = pd.concat([file_name_df, basic_feature_df, trans_feature_df, cross_feature_df])
    # fill all the na elements rows with 0
    allFeatures = allFeatures.fillna(0)
    return allFeatures


def columbusSDK(wavPath=None, asrDf=None):
    feature_result = single_file_processor(g_config, wavPath, asrDf)
    if feature_result is None:
        print('get feature fail!')

    return feature_result


def get_result_from_ali(result_dict):
    """

    :param result_dict: asr识别结果，已经改成了实时化asr结果的dict.
    :return: pd处理结果， asr识别json
    """
    json_dict = result_dict
    ret = {"sentence_id": [],
           "begin_time": [],
           "end_time": [],
           "text": [],
           "status_code": []
           }
    asr_json = {"result": []}
    idx = 1
    for item in json_dict['result']:
        # 这里出现的or主要针对新旧两个asr的字段不同而设计
        ret["sentence_id"].append(item["sentence_id"])
        ret["status_code"].append(item["status_code"])
        ret["begin_time"].append(item["begin_time"])
        ret["end_time"].append(item["end_time"])
        ret["text"].append(item["text"])

        asr_json["result"].append(item)
        idx += 1

    pd_result = pd.DataFrame(ret, columns=['sentence_id', 'begin_time',
                                           'end_time', 'status_code', 'text'])

    # 针对新asr可能的bug。
    # by 郝洋
    # 2020年02月04日
    last_row_num = pd_result.shape[0]
    try:
        if pd_result.iloc[last_row_num - 1]['begin_time'] == 0:
            pd_result = pd_result.iloc[:last_row_num - 1]
    except:
        pass

    # print(pd_result, '\n\n\n\n\n')
    return pd_result, asr_json




def get_result_from_raw(result_dict):
    """

    :param result_dict: asr识别结果，已经改成了实时化asr结果的dict.
    :return: pd处理结果， asr识别json
    """
    json_dict = result_dict
    ret = {"sentence_id": [],
           "begin_time": [],
           "end_time": [],
           "text": [],
           "status_code": []
           }
    asr_json = {"result": []}
    idx = 1
    for item in json_dict['result']:
        # 这里出现的or主要针对新旧两个asr的字段不同而设计
        ret["sentence_id"].append(item.get("index") or idx)
        ret["status_code"].append(item.get("code") or 0)
        ret["begin_time"].append(item.get("beginTime") or item.get("begin_time") or 0)
        ret["end_time"].append(item.get("endTime") or item.get("end_time"))
        ret["text"].append(item.get("result") or item.get("text"))

        tmp = {
            "sentence_id": item.get("index") or idx,
            "status_code": item.get("code") or 0,
            "begin_time": item.get("beginTime") or item.get("begin_time"),
            "end_time": item.get("endTime") or item.get("end_time"),
            "text": item.get("result") or item.get("text")
        }
        asr_json["result"].append(tmp)
        idx += 1

    pd_result = pd.DataFrame(ret, columns=['sentence_id', 'begin_time',
                                           'end_time', 'status_code', 'text'])

    # 针对新asr可能的bug。
    # by 郝洋
    # 2020年02月04日
    last_row_num = pd_result.shape[0]
    try:
        if pd_result.iloc[last_row_num - 1]['begin_time'] == 0:
            pd_result = pd_result.iloc[:last_row_num - 1]
    except:
        pass

    # print(pd_result, '\n\n\n\n\n')
    return pd_result, asr_json


def removeCharacter(item):
    import re
    item = re.sub(r'[^\w\s]', '', item)
    item = item.replace(' ', '')
    return len(item)


def getAbsTime(item):
    item = item // 1000
    return "{:0>2}:{:0>2}:{:0>2}".format(item // 3600, int(item // 60 % 60), int(item % 60))


def mergeResultAndSave(textdata, result_json, convert=True):
    """
    :param textdata:
    :param result_json:
    :param convert:
    :return: 返回asr识别json
    """
    file, asr_json = get_result_from_ali(result_json)
    if file.shape[0] == 0:
        begin_time = 0
        end_time = 0
        text = ''
        len_time = 0
        len_text = 0
        abs_begin_time = getAbsTime(begin_time // 1000)
        abs_end_time = getAbsTime(end_time // 1000)
        data = {'sentence_id': 1, 'begin_time': [begin_time], 'end_time': [end_time], 'text': [text],
                'sentence_time_length': [len_time], 'text_length': [len_text],
                'abs_begin_time': [abs_begin_time], 'abs_end_time': [abs_end_time]}
        pd_result = pd.DataFrame(data)
    elif len(file.text.sum()) == 0:
        begin_time = 0
        end_time = int(file.iloc[-1].end_time)
        text = ''
        len_time = end_time - begin_time
        len_text = 0
        abs_begin_time = getAbsTime(begin_time // 1000)
        abs_end_time = getAbsTime(end_time // 1000)
        data = {'sentence_id': 1, 'begin_time': [begin_time], 'end_time': [end_time], 'text': [text],
                'sentence_time_length': [len_time], 'text_length': [len_text],
                'abs_begin_time': [abs_begin_time], 'abs_end_time': [abs_end_time]}
        pd_result = pd.DataFrame(data)
    else:
        # 计算长度并将文件写入一行向量
        file.end_time = file.end_time.astype(int)
        file.begin_time = file.begin_time.astype(int)

        max_len = file.iloc[-1].end_time + 1
        map_array = np.zeros((max_len)) - 1
        for index in range(file.shape[0]):
            i = file.iloc[index]
            if len(i.text) != 0:
                map_array[i.begin_time:i.end_time] = index + 1
        # 融合文件并严格对齐
        result = {'begin_time': [], 'end_time': [], 'text': []}
        begin = 0
        end = 0
        change = False
        lastValue = map_array[0]
        for i in range(1, map_array.shape[0]):
            try:
                if lastValue == map_array[i]:
                    continue
                sentence_id = lastValue
                lastValue = map_array[i]
                if sentence_id == -1:
                    content = ''
                    result['begin_time'].append(begin + 1)
                    result['end_time'].append(i - 1)
                else:
                    content = file.iloc[int(sentence_id - 1)].text
                    result['begin_time'].append(begin)
                    result['end_time'].append(i)
                result['text'].append(content)
                textdata.append(content)
                begin = i
                lastValue = map_array[i]
            except:
                pass
        # print(result)
        result['begin_time'][0] = 0
        pd_result = pd.DataFrame(result)
        pd_result['sentence_time_length'] = \
            pd_result.apply(lambda x: x.end_time - x.begin_time, axis=1) / 1000
        pd_result['text_length'] = pd_result.text.apply(removeCharacter)
        pd_result['abs_begin_time'] = pd_result.begin_time.apply(getAbsTime)
        pd_result['abs_end_time'] = pd_result.end_time.apply(getAbsTime)
        pd_result['sentence_id'] = np.arange(1, pd_result.shape[0] + 1)
    if convert:
        pd_result = convert_to_glb(pd_result)
    return pd_result, asr_json


def convert_to_glb(pd_result):
    pd_result['sentence_time_length'] = pd_result.end_time - pd_result.begin_time
    columns = {'sentence_id': 'index', 'sentence_time_length': 'timeLength', 'text': 'text',
               'text_length': 'textLength'}
    pd_result = pd_result.rename(columns=columns)
    return pd_result[['index', 'timeLength', 'textLength', 'text']]


def get_columbus_feature(asr_json: dict, wav_path: str):
    a, b = mergeResultAndSave([], asr_json)
    a = a.replace('', np.nan)
    # print(a, '\n\n\n')
    ret = columbusSDK(wavPath=wav_path, asrDf=a)
    return ret


if __name__ == '__main__':
    asr = """{
        "result": [{"end_time": 4420, "status_code": 0, "sentence_id": 1, 
        "begin_time": 0, "text": "这里是3所以这里是四因为。"},
                   {"end_time": 5690, "status_code": 0, "sentence_id": 2, "begin_time": 4620, "text": "嗯"},
                   {"end_time": 11590, "status_code": 0, "sentence_id": 3, "begin_time": 5800,
                    "text": "相对的两个面加起来的和17"},
                   {"end_time": 22550, "status_code": 0, "sentence_id": 4, "begin_time": 11940,
                    "text": "他，他是四的话，那自己是也是四因为这两个相贴的数加起来，必须要等等于8"},
                   {"end_time": 30340, "status_code": 0, "sentence_id": 5, "begin_time": 23350,
                    "text": "但这里是孙呢，对的，这里就是3所以这个问号就是3"}]}"""
    x = get_columbus_feature(asr, '../aaa.wav')
    print(x)
