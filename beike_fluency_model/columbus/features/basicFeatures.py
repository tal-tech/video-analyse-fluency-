#!/usr/bin/python
#encoding:utf-8
'''
Project 'Columbus' basic feature function modules
author: Guowei Xu
Date: July 11, 2018
Modified: 08/02/2018

'''
import pandas as pd
import numpy as np
import wave
import jieba
import requests
import json
import jieba.posseg as pseg
import time
import re

#cache是一个全局变量贮存已经计算过的数值，避免重复运算
def openFile(path, cache):
    df = pd.read_excel(path)
    cache['df'] = df
    return df

#将句子拼接并返回所有文字
def getText(df, cache):
    text = ''
    for t in df['text']:
         if(isinstance(t, str)):#检查是否为string，排除nan
            text+=t.replace(" ", '')
    cache['text'] = text
    return text

def cacheDF(df, cache):
    cache['df'] = df
    return df



#说话时长(秒)
def getVoiceLen(df, cache):
    # voiceLen = float(sum(df['timeLength']))/1000
    voiceLen = float(np.sum(df[df['text'].notnull()]['timeLength']))/1000
    cache['voiceLen'] = voiceLen
    return voiceLen



#文件时长
####################################################3
######### This function  is from 心汝
#####################################################
def getFileLen(path_to_wav, cache):
    #open a wave file, and return a Wave_read object
    f = wave.open(path_to_wav,"rb")
    #read the wave's format infomation,and return a tuple
    params = f.getparams()
    #get the info
    framerate, nframes = params[2:4]
    fileLen = nframes * (1.0/framerate)
    cache['fileLen'] = fileLen
    return fileLen


#语音语速
def getSpeedByVoice(df, cache):
    totalCharNum, voiceLen = 0.0, 0.0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'voiceLen' in cache.keys()):
        voiceLen = getVoiceLen(df, cache)
    else:
        voiceLen = cache['voiceLen']
    if(voiceLen != 0):
        speedByVoice = totalCharNum/voiceLen #总字数处以总说话时长
        cache['speedByVoice'] = speedByVoice
    else:
        speedByVoice = 0
    return speedByVoice

#文件语速, 字数/文件时长
def getSpeedByFile(df, path_to_wav, cache):
    totalCharNum, fileLen = 0.0, 0.0
    if(not('totalCharNum' in cache.keys())):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not('fileLen' in cache.keys())):
        fileLen =  getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    return totalCharNum/fileLen


#总字数
def getTotalCharNum(df, cache):
    if not 'text' in cache.keys():
        text = getText(df,cache)
    else:
        text = cache['text']
    totalCharNum = float(len(re.findall(r'\w', text)))
    # totalCharNum = float(sum(df['textLength']))
    cache['totalCharNum'] = totalCharNum
    return totalCharNum


#有效说话百分比
def getVoiceOverFilePercent(df, path_to_wav, cache):
    fileLen, voiceLen = 0.0, 0.0
    if(not 'fileLen' in cache.keys()):
        fileLen = getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    if(not 'voiceLen' in cache.keys()):
        voiceLen = getVoiceLen(df, cache)
    else:
        voiceLen = cache['voiceLen']
    voiceOverFilePercent = float(voiceLen)/fileLen
    cache['voiceOverFilePercent'] = voiceOverFilePercent
    return voiceOverFilePercent


#总句数
def getTotalSentNum(df, cache):
    # return df['text'].shape[0]
    return np.sum(df['text'].notnull())


#平均句长
def getAvgCharPerSent(df, cache):
    totalCharNum, charNum = 0.0, 0.0
    if(not ('totalCharNum' in cache.keys() and 'charNum' in cache.keys())):
        totalCharNum= getTotalCharNum(df, cache)
        charNum = getCharNum(df, cache)
        avgCharPerSent = float(totalCharNum)/charNum.shape[0]
    else:
        avgCharPerSent = float(cache['totalCharNum'])/cache['charNum'].shape[0]
    cache['avgCharPerSent'] = avgCharPerSent
    return avgCharPerSent

#总疑问句
#目前是统计整个文本包含多少问号，方法比较原始，后期可做改进
def getTotalQuestionSentNum(df, cache):
    if not 'text' in cache.keys():
        text = getText(df,cache)
    else:
        text = cache['text']
    return text.count('？')+text.count('?')#这里两个问号不同，前一个事中文输入法后一个为英文输入法

#计算词性并把分词按照词性分为不同的list，放到cache里
def getPOS(text, cache):
    noun = []
    verb = []
    adj = []
    adv = []
    noun_flag = ['n', 's', 'nr', 'ns', 'nt', 'nw', 'nz', 'vn']
    verb_flag = ['v', 'vd', 'vn']
    adj_flag = ['a', 'ad', 'an']
    adv_flag = ['d']
    words = pseg.cut(text)
    for w in words:
        if(w.flag in noun_flag):
            noun.append(w)
        elif(w.flag in verb_flag):
            verb.append(w)
        elif(w.flag in adj_flag):
            adj.append(w)
        elif(w.flag in adv_flag):
            adv.append(w)
    cache['noun'] = noun
    cache['verb'] = verb
    cache['adj'] = adj
    cache['adv'] = adv
    return noun, verb, adj, adv

#总名词数量
def getTotalNounNum(df, cache):
    # changed to df
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    nounNum = 0
    if(not 'noun' in cache.keys()):
        noun, _, _, _ = getPOS(text, cache)
        nounNum = len(noun)
    else:
        nounNum = len(cache['noun'])
    return nounNum

#总动词数量
def getTotalVerbNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    verbNum = 0
    if(not 'verb' in cache.keys()):
        _, verb, _, _ = getPOS(text, cache)
        verbNum = len(verb)
    else:
        verbNum = len(cache['verb'])
    return verbNum

#总形容词数量
def getTotalAdjNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    adjNum = 0
    if(not 'adj' in cache.keys()):
        adj = getPOS(text, cache)
        adjNum = len(adj)
    else:
        adjNum = len(cache['adj'])
    return adjNum

#总副词数量
def getTotalAdvNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    advNum = 0
    if(not 'adv' in cache.keys()):
        adv = getPOS(text, cache)
        advNum = len(adv)
    else:
        advNum = len(cache['adv'])
    return advNum

#获取X出现的次数，需要提供包含X词的list X_list
def getTotalXNum(df, X_list, cache):
    totalXNum = np.zeros(len(X_list))
    text = ""
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    for i in range(len(X_list)):
        totalXNum[i] = countXNum(text, X_list[i])
    totalXNum = np.sum(totalXNum)
    cache['totalXNum'] = totalXNum
    return totalXNum

#以滑窗的形式遍历text，统计某个词x出现的次数，滑窗长度和x一样
def countXNum(text, x):
    count = 0
    windowSize = len(x)
    for i in range(0, len(text)-windowSize+1):
        if(text[i:i+windowSize] == x):
            count += 1
    return count



#正情感词数目
def getTotalPosNum(df, pauseword, cache):
    if(not 'totalPosNum' in cache.keys()):
        totalPosNum, totalNegNum = sentimentJudger(df, pauseword, cache)
        cache['totalPosNum'] = totalPosNum
        cache['totalNegNum'] = totalNegNum
        return totalPosNum
    else:
        return cache['totalPosNum']

#负情感词数目
def getTotalNegNum(df, pauseword, cache):
    if(not 'totalNegNum' in cache.keys()):
        totalPosNum, totalNegNum = sentimentJudger(df, pauseword, cache)
        cache['totalPosNum'] = totalPosNum
        cache['totalNegNum'] = totalNegNum
        return totalNegNum
    else:
        return cache['totalNegNum']


#统计词语的情感，忽略停顿词
def sentimentJudger(df, pauseword, cache):
    text = ""
    pos_count = 0
    neg_count = 0
    AK = 'ONltuGBOpjucDTBrO1XlkKK9'
    SK = 'aPkeHTSHk0LN325BBClimROiRiYceaFx'
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    au_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(
    AK, SK)
    access_token = requests.post(au_url,verify=False).json()['access_token']
    headers = {'Content-Type': 'application/json'}
    api = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={}'.format(access_token)
    
    if(not'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    words = pseg.cut(text)
    for w in words:
        if(w in pauseword):
            continue
        data={"text": str(w)}
        result = requests.post(api,data=json.dumps(data),headers=headers).json()
        time.sleep(0.2)
        sentiment = result['items'][0]['sentiment']
        if(sentiment==2):
            pos_count +=1
        elif(sentiment ==0):
            neg_count +=1
    cache['totalPosNum'] = pos_count
    cache['totalNegNum'] = neg_count
    return pos_count, neg_count



#总停顿词数目
def getTotalPauseWordNum(df, pauseword, cache):
    totalPauseWordNum = np.zeros(len(pauseword))
    text = ""
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    for i in range(len(pauseword)):
        totalPauseWordNum[i] = countXNum(text, pauseword[i])
    cache['totalPauseWordNum'] = totalPauseWordNum
    return sum(totalPauseWordNum)


#每句话时长(秒)
def getSentLen(df, cache):
    sentLen = np.array(df[df['text'].notnull()]['timeLength'], dtype=np.float64)/1000
    cache['sentLen'] = sentLen
    return sentLen


#每句话字数
def getCharNum(df, cache):
    charNum = np.array([len(re.findall(r'\w',re.sub('_','',str(x)))) for x in df[df['textLength']>0]['text']],  dtype=np.float64)

    cache['charNum'] = charNum
    return charNum


#每句话语速
def getSentSpeed(df, cache):
    sentLen, charNum = 0.0, 0.0
    if(not ('sentLen') in cache.keys()):
        sentLen = getSentLen(df, cache)
    else:
        sentLen = cache['sentLen']
    if(not ('charNum' in cache.keys())):
        charNum = getCharNum(df, cache)
    else:
        charNum = cache['charNum']
    sentSpeed = charNum/sentLen
    cache['sentSpeed']= sentSpeed
    return sentSpeed



# New Features
def getAvgSpeed(df, cache):
    avgSpeed = 0.0
    if(not 'sentSpeed' in cache.keys()):
        sentSpeed = getSentSpeed(df, cache)
        avgSpeed = sum(sentSpeed)/sentSpeed.shape[0]
    else:
        sentSpeed = cache['sentSpeed']
        avgSpeed = sum(sentSpeed)/sentSpeed.shape[0]
    cache['avgSpeed'] = avgSpeed
    return avgSpeed

#以2秒停顿为分句，每个短句字数方差
def getCharNumVar(df, cache):
    charNumVar = 0.0
    if(not (('avgCharNum' in cache.keys() and 'charNum' in cache.keys()))):
        avgCharNum = getAvgCharPerSent(df, cache)
        charNum = getCharNum(df, cache)
        charNumVar = sum(map(lambda x: (x-avgCharNum)*(x-avgCharNum), charNum))/charNum.shape[0]
    else:
        charNumVar = sum(map(lambda x: (x-cache['avgCharPerSent'])*(x-cache['avgCharPerSent']), cache['charNum']))/cache['charNum'].shape[0]

    cache['charNumVar'] = charNumVar
    return charNumVar


#以2秒停顿为分句，每个短句的语速方差
def getSpeedVar(df, cache):
    speedChar = 0.0
    if(not ('avgSpeed' in cache.keys() and 'sentSpeed' in cache.keys())):
        avgSpeed = getAvgSpeed(df, cache)
        sentSpeed = getSentSpeed(df, cache)
        speedVar =sum(map(lambda x: (x-avgSpeed)*(x-avgSpeed), sentSpeed))/sentSpeed.shape[0]
    else:
        speedVar =sum(map(lambda x: (x-cache['avgSpeed'])*(x-cache['avgSpeed']), cache['sentSpeed']))/cache['sentSpeed'].shape[0]
    cache['speedVar'] = speedVar
    return speedVar


def getDuplicate1Abs(df, cache):
    #只计算重复的连续
    N=1
    text = ""
    totalCharNum = 0
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    repeat_num = 0
    two_gram_dict = []
    for i in range(len(text)):
        two_gram_dict.append([text[i:i+N]])
    for index,value in enumerate(two_gram_dict):
        if index < len(two_gram_dict)-1:
            if two_gram_dict[index] == two_gram_dict[index + 1]:
                if not str.isdigit(str(two_gram_dict[index])): #去除数字
                    repeat_num += 1
    cache['repeat_num_1'] = repeat_num
    return repeat_num

#重复一个字占所有字数百分比
# 基于张文矜代码有修改
def getDuplicate1Percent(df, cache):
    #只计算重复的连续
    totalCharNum, repeat_num_1 = 0, 0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'repeat_num_1' in cache.keys()):
        repeat_num_1 = getDuplicate1Abs(df, cache)
    else:
        repeat_num_1 = cache['repeat_num_1']
    if(totalCharNum!=0):
        return repeat_num_1/totalCharNum
    else:
        return 0

def getDuplicate2Abs(df, cache):
    #只计算重复的连续
    N=2
    text = ""
    totalCharNum = 0
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    repeat_num = 0
    two_gram_dict = []
    for i in range(len(text)):
        two_gram_dict.append([text[i:i+N]])
    for index,value in enumerate(two_gram_dict):
        if index < len(two_gram_dict)-1:
            if two_gram_dict[index] == two_gram_dict[index + 1]:
                if not str.isdigit(str(two_gram_dict[index])): #去除数字
                    repeat_num += 1
    cache['repeat_num_2'] = repeat_num
    return repeat_num

#重复两个字所占百分比
#基于张文矜代码有修改
def getDuplicate2Percent(df, cache):
    #只计算重复的连续
    totalCharNum, repeat_num_2 = 0, 0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'repeat_num_2' in cache.keys()):
        repeat_num_2 = getDuplicate2Abs(df, cache)
    else:
        repeat_num_2 = cache['repeat_num_2']
    if(totalCharNum!=0):
        return repeat_num_2/totalCharNum
    else:
        return 0

# Here are the new features for Sea project
def getLongSentPercent(df, cache):
    if 'charNum' not in cache.keys():
        charNum = getCharNum(df, cache)
    else:
        charNum = cache['charNum']
    long_sent_percent = np.sum(charNum > np.mean(charNum))/len(charNum)
    cache['long_sent_percent'] = long_sent_percent
    return long_sent_percent

def getEnWordPercent(df, cache):
    clean_sent = [re.findall(r'\w',str(x)) for x in df[df['textLength']>0]['text']]
    enWordpercent = np.sum(pd.Series([x == ['嗯'] for x in clean_sent]))/len(clean_sent)
    cache['en_word_percent'] = enWordpercent
    return enWordpercent

def getLongBlankNum(df, cache):
    inner_df = df.copy()
    inner_df['group_mark'] = inner_df['text'].isnull()
    inner_df['group_mark'] = inner_df['group_mark'].apply(lambda x: 0 if x else 1)
    inner_df['group_mark'] = np.cumsum(inner_df['group_mark']) + (1 - inner_df['group_mark'])*inner_df.shape[0]
    allsent_count = inner_df.groupby('group_mark').apply(lambda x: x.shape[0])
    emptysent_count = allsent_count[np.sum(inner_df['text'].notnull()):]
    LongBlankCount = np.sum(emptysent_count >= 2)
    cache['LongBlankNum'] = LongBlankCount
    return LongBlankCount

def getLongChatNum(df, cache):
    inner_df = df.copy()
    inner_df['group_mark'] = inner_df['text'].notnull()
    inner_df['group_mark'] = inner_df['group_mark'].apply(lambda x: 0 if x else 1)
    inner_df['group_mark'] = np.cumsum(inner_df['group_mark']) + (1 - inner_df['group_mark'])*inner_df.shape[0]
    allsent_count = inner_df.groupby('group_mark').apply(lambda x: x.shape[0])
    chatsent_count = allsent_count[np.sum(inner_df['text'].isnull()):]
    LongChatCount = np.sum(chatsent_count >= 2)
    cache['LongChatNum'] = LongChatCount
    return LongChatCount

# define the helper function to find out total talk length
def getAliLenPercent(df, path_to_wav, cache):
    fileLen, AliLen = 0.0, 0.0
    if(not('fileLen' in cache.keys())):
        fileLen =  getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    AliLen = float(np.sum(df['timeLength']))/1000
    cache['AliLen'] = AliLen
    AliLenPercent = AliLen/fileLen
    return AliLenPercent

# this line help us load the word list files into our work space
def getWordList(path_to_words):
    with open(path_to_words, 'r', encoding='utf-8') as word_file:
        word_list = word_file.readlines()
    word_file.close()
    word_list = [re.sub(r'[\n|，|,| ]', '', x) for x in word_list]
    return word_list
        

# start from this line, we will rewrite the basic functions and make sure the return value of these functions
# are at the sentence level
def getSentLenVector(df, cache):
    sent_len_vector = df['timeLength'].values / 1000
    cache['SentLenVector'] = sent_len_vector
    return sent_len_vector

def getCharNumVector(df, cache):
    temp = df['text'].fillna('')
    char_num_vector = np.array([len(re.findall(r'\w',x)) for x in temp],  dtype=np.float64)
    cache['CharNumVector'] = char_num_vector
    return char_num_vector

def getSentSpeedVector(df, cache):
    if 'SentLenVector' not in cache.keys():
        sent_len_vector = getSentLenVector(df, cache)
    else:
        sent_len_vector = cache['SentLenVector']
    if 'charNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # since the unit for time is ms, we need to times 1000 to the nominator
    sent_speed_vector = char_num_vector/sent_len_vector
    cache['SentSpeedVector'] = sent_speed_vector
    return sent_speed_vector

def getQuestionSentNumVector(df, cache):
    # here we still keep the same logic as total sentnum function
    temp = df['text'].fillna('')
    question_sentnum_vector = temp.apply(lambda x: x.count('？')+x.count('?')).values
    return question_sentnum_vector

# This line is used to cut the words for each sentence
def getPOSVector(df, cache):
    temp = df['text'].fillna('')
    pos = [getPOS(x, dict()) for x in temp]
    cache['POS'] = pos
    return pos

def getNounNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    noun_num_vector = [len(x[0]) for x in pos]
    return noun_num_vector

def getVerbNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    verb_num_vector = [len(x[1]) for x in pos]
    return verb_num_vector

def getAdjNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    adj_num_vector = [len(x[2]) for x in pos]
    return adj_num_vector

def getAdvNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    adv_num_vector = [len(x[3]) for x in pos]
    return adv_num_vector

def getPauseWordNumVector(df, pauseword, cache):
    temp = df['text'].fillna('')
    pause_wordnum_vector = temp.apply(lambda x: np.sum([countXNum(x, y) for y in pauseword])).values
    return pause_wordnum_vector

def getXNumVector(df, X_list, cache):
    temp = df['text'].fillna('')
    xnum_vector = temp.apply(lambda x: np.sum(countXNum(x, y) for y in X_list)).values
    return xnum_vector

# this small function is a helper function to find out two consecutive identical words
def consecutiveWordCount(text, n):
    # first step, filter out any number or character
    temp = re.sub(r'[^\w]|_|\d','', text)
    # based on the length of temp we need to go through the whole string
    consecutive_word_count = np.sum([temp[x:x+n]==temp[x+1:x+n+1] for x in range(len(temp)-1)])
    return consecutive_word_count

def getDuplicate1AbsVector(df, cache):
    temp = df['text'].fillna('')
    dup_1abs_vector = temp.apply(lambda x: consecutiveWordCount(x, 1)).values
    cache['Duplicate1AbsVector'] = dup_1abs_vector
    return dup_1abs_vector

def getDuplicate1PercentVector(df, cache):
    if 'Duplicate1AbsVector' not in cache.keys():
        dup_1abs_vector = getDuplicate1AbsVector(df, cache)
    else:
        dup_1abs_vector = cache['Duplicate1AbsVector']
    if 'CharNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # fill infinite with 0
    dup_1percent_vector = dup_1abs_vector/char_num_vector
    dup_1percent_vector[np.isinf(dup_1percent_vector)] = 0
    dup_1percent_vector[np.isnan(dup_1percent_vector)] = 0
    return dup_1percent_vector

def getDuplicate2AbsVector(df, cache):
    temp = df['text'].fillna('')
    dup_2abs_vector = temp.apply(lambda x: consecutiveWordCount(x, 2)).values
    cache['Duplicate2AbsVector'] = dup_2abs_vector
    return dup_2abs_vector

def getDuplicate2PercentVector(df, cache):
    if 'Duplicate2AbsVector' not in cache.keys():
        dup_2abs_vector = getDuplicate2AbsVector(df, cache)
    else:
        dup_2abs_vector = cache['Duplicate2AbsVector']
    if 'CharNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # fill infinite with 0
    dup_2percent_vector = dup_2abs_vector/char_num_vector
    dup_2percent_vector[np.isinf(dup_2percent_vector)] = 0
    dup_2percent_vector[np.isnan(dup_2percent_vector)] = 0
    return dup_2percent_vector

# Here we may add 8 more features corrresponding to the version 1.2's feature

def getThreeTimeLess(df, cache):
    # before we do here, we need to find only the words in a sentence
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    three_time_less = np.sum([x > 0 and x < 3 for x in char_num])
    cache['three_time_less'] = three_time_less
    return three_time_less

def getBetweenThreeTen(df, cache):
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    between_three_ten = np.sum([x >= 3 and x <= 10 for x in char_num])
    cache['between_three_ten'] = between_three_ten
    return between_three_ten

def getTenTimeMore(df, cache):
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    ten_time_more = np.sum([x > 10 for x in char_num])
    cache['ten_time_more'] = ten_time_more
    return ten_time_more
    

#
#
# def getMaxMinSentLen(df, cache):
#     maxSentLen, minSentLen = 0, 0
#     if(not ('textLength') in cache.keys()):
#         _, textLength = getTextLength(df)
#         maxSentLen, minSentLen = max(textLength), min(textLength)
#     else:
#         maxSentLen, minSentLen = max(cache['textLength']), min(cache['textLength'])
#     cache['maxSentLen'] = maxSentLen
#     cache['minSentLen'] = minSentLen
#     return maxSentLen, minSentLen
#
# def getVarianceSentLen(df, cache):
#     varianceSentLen = 0.0
#     if(not ('avgSentLen' in cache.keys() and 'textLength' in cache.keys())):
#         avgSentLen = getAvgSentLen(df)
#         _, textLength = getTextLength(df)
#         varianceSentLen = sum(map(lambda x: (x-avgSentLen)*(x-avgSentLen), textLength))/textLength.shape[0]
#     else:
#         varianceSentLen = sum(map(lambda x: (x-cache['avgSentLen'])*(x-cache['avgSentLen']), cache['textLength']))/cache['textLength'].shape[0]
#     cache['varianceSentLen'] = varianceSentLen
#     return varianceSentLen
