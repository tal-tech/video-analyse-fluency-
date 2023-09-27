#coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))

import pandas as pd
from columbus import get_columbus_feature


COLS = ['AvgCharPerSent', 'AvgSpeed', 'CharNumVar', 'CharNum_25Per',
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


def batch_extract(input_asrs, input_wavs):
    """
    :param input_asrs: dict. key-clip_name, asr_result
    :param input_wavs: dict. key-clip_name, value-clip_path
    :return:
    """

    results = {}

    for clip in input_wavs:

        asr = input_asrs[clip]
        wav = input_wavs[clip]

        columbus_result = get_columbus_feature(asr, wav)
        columbus_df = pd.DataFrame([columbus_result], columns=COLS)

        columbus_feature = columbus_df.values

        results[clip] = columbus_feature

    return results


