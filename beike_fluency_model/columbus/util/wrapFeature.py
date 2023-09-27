#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Harvey
Date: 07/12/2018
Modified: 08/02/2018
Including: three wrap-up function for all needed feature
functions following layered structure
'''

import sys
import pandas as pd
from itertools import combinations
import numpy as np
import re
import xlrd

sys.path.append('..')
from columbus.features.basicFeatures import *
from columbus.features.transFeatures import *
from columbus.features.crossFeatures import *


def getBasicFeatures(configObj, wavPath, asrDf):
    # txt_file_path = configObj['input']['path'] + file_name
    # Here we suppose files are in the same directory
    wav_file_path = wavPath
    cache = {}
    result = []
    basic_features = [key for key, value in configObj['basicFeatures'].items() if int(value) == 1]
    result_column = []
    # df = openFile(txt_file_path, cache)
    df = cacheDF(asrDf, cache)
    # Here we make the justification
    if np.sum(df['text'].notnull()) == 0:
        # at this line we need to generate the feature names by our selves
        stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
        basic_features = [x for x in basic_features if x not in stat_features]
        stat_features = [y.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var'] for y in stat_features]
        result_column = [x.split('get')[1] for x in basic_features] + stat_features
        return pd.Series(index = result_column), cache.copy()
    if 'getFileLen' in basic_features:
        result += [getFileLen(wav_file_path, cache)]
        result_column += ['FileLen']
    if 'getSpeedByFile'in basic_features:
        result += [getSpeedByFile(df,wav_file_path,cache)]
        result_column += ['SpeedByFile']
    if 'getVoiceOverFilePercent' in basic_features:
        result += [getVoiceOverFilePercent(df, wav_file_path, cache)]
        result_column += ['VoiceOverFilePercent']
    if 'getTotalXNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['wordListPath'])
        result += [getTotalXNum(df, wordList, cache)]
        result_column += ['TotalXNum']
    if 'getTotalPauseWordNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalPauseWordNum(df, wordList, cache)]
        result_column += ['TotalPauseWordNum']
    if 'getTotalPosNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalPosNum(df, wordList, cache)]
        result_column += ['TotalPosNum']
    if 'getTotalNegNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalNegNum(df, wordList, cache)]
        result_column += ['TotalNegNum']
    if 'getAliLenPercent' in basic_features:
        result += [getAliLenPercent(df, wav_file_path, cache)]
        result_column += ['AliLenPercent']
    # Here we group up the three statistic features
    stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
    for func in stat_features:
        temp_result = eval(func + '(df,cache)')
        result += [np.mean(temp_result),np.std(temp_result)] + list(pd.Series(temp_result).describe())[3:] + [np.var(temp_result)]
        result_column += [func.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var']]
    exclude_features = ['getFileLen','getSpeedByFile','getVoiceOverFilePercent','getTotalXNum','getTotalPauseWordNum','getTotalPosNum','getTotalNegNum','getAliLenPercent'] + stat_features
    basic_features = [x for x in basic_features if x not in exclude_features]
    for func in basic_features:
        result += [eval(func + '(df, cache)')]
    result_column += [x.split('get')[1] for x in basic_features]
    return pd.Series(result, index=result_column), cache.copy()

def getTransFeatures(configObj, basic_feature_df, cache):
    trans_features = pd.Series([key for key, value in configObj['transFuncs'].items() if int(value) == 1])
    if len(trans_features) == 0:
        return pd.Series()
    trans_columns = [x+'_'+y.split('get')[1] for y in trans_features for x in basic_feature_df.index]
    # add one more line to justify whether all elements in basic features are NA
    if np.sum(basic_feature_df.notnull()) == 0:
        return pd.Series(index = trans_columns)
    trans_result = []
    for func in trans_features:
        trans_result += [eval(func+'(basic_feature_df)')]
    # trans_result = pd.concat(trans_result)
    # trans_columns = [x+'_'+y.split('get')[1] for y in trans_features for x in basic_feature_df.index]
    return pd.Series(list(pd.concat(trans_result)), index = trans_columns)

def getCrossetCrossFeatures(configObj, basic_feature_df, cahce):
    cross_features = [item[0] for item in configObj['crossFuncs'].items() if len(item[1]) !=0]
    cross_result = []
    cross_column = []
    for func in cross_features:
        feature_pairs = configObj['crossFuncs'][func]
        if feature_pairs == ['ALL'] or feature_pairs == ['All']:
            basic_features = [key.split('get')[1] for key, value in configObj['basicFeatures'].items() if int(value) == 1]
            feature_pairs = list(combinations(basic_features, 2))
        for pair in feature_pairs:
            cross_result += [eval('get'+func+'(basic_feature_df["'+pair[0]+'"],basic_feature_df["'+pair[1]+'"])')]
            cross_column += [pair[0]+'_'+pair[1]+'_'+func]
    if np.sum(basic_feature_df.notnull()) == 0:
        return pd.Series(index=cross_column)
    return pd.Series(list(cross_result), index=cross_column)

# design an specific function used for sentence level feature
def getSentenceFeatures(configObj, asrDf):
    # txt_file_path = configObj['input']['path'] + file_name
    cache = {}
    result = []
    basic_features = [key for key, value in configObj['basicFeatures'].items() if int(value) == 1]
    result_column = []
    # df = openFile(txt_file_path, cache)
    df = cacheDF(asrDf, cache)
    # filter out the featurs that we can do nothing with
    available_sentence_feature = ['getSentLen','getCharNum','getSentSpeed','getDuplicate1Abs','getDuplicate1Percent',
    'getDuplicate2Abs','getDuplicate2Percent','getTotalQuestionSentNum','getTotalNounNum','getTotalVerbNum','getTotalAdjNum',
    'getTotalAdvNum','getTotalXNum','getTotalPauseWordNum']
    basic_features = list(set(basic_features).intersection(set(available_sentence_feature)))
    # change some string part of basic_features
    basic_features = [x.replace('Total','')+ 'Vector' for x in basic_features]
    # after we do this we can run the basic features one by one
    for func in basic_features:
        result_column += [re.sub('get|Vector', '', func)]
        if np.sum(df['text'].notnull()) == 0:
            continue
        if func == 'getPauseWordNumVector':
            wordList = getWordList(configObj['wordInput']['stopWordsPath'])
            result += [eval(func + "(df, wordList, cache)")]
        elif func == 'getXNumVector':
            wordList = getWordList(configObj['wordInput']['wordListPath'])
            result += [eval(func + "(df, wordList, cache)")]
        else:
            result += [eval(func + '(df, cache)')]
    result = np.array(result).T
    return result, result_column