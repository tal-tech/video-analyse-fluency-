# coding=utf-8
import os
import sys

base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))

import shutil
import pickle
import numpy as np
import tensorflow as tf
import json
import uuid
import opensmile_extraction_fluency
import extract_columbus_features

__all__ = ["audio_fluency_model"]


def print_error(msg):
    print("\033[31;1m{}\033[0m".format(msg))


class Fluency:

    def __init__(self, rll_model_path, lr_model_path):
        self.dimension = 1632
        self.graph, self.session = self.reloadGraph(rll_model_path)
        self.X = tf.placeholder(tf.float32, shape=[None, self.dimension], name='input')

        # with loaded_sess as sess:
        w1 = self.graph.get_tensor_by_name('fc_l1/weights:0')
        b1 = self.graph.get_tensor_by_name('fc_l1/biases:0')
        w2 = self.graph.get_tensor_by_name('fc_l2/weights:0')
        b2 = self.graph.get_tensor_by_name('fc_l2/biases:0')
        self.embd = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(self.X, w1) + b1), w2) + b2)

        self.lr_model = pickle.load(open(lr_model_path, 'rb'))

    def inference(self, inputX):
        feed = {self.X: inputX}
        output = self.session.run(self.embd, feed_dict=feed)
        return output

    def reloadGraph(self, modelPath):
        tf.reset_default_graph()
        sess = tf.Session()

        metaFile = modelPath.split('/')[-1] + '.ckpt.meta'
        saver = tf.train.import_meta_graph(os.path.join(modelPath, metaFile))
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))
        graph = tf.get_default_graph()
        return graph, sess

    def predict(self, wav_clips_list=None, asr_results=None):
        '''
        :param wav_clips_list: List[{"wav_path":"path/to/your/clips/1.wav", "begin_time":0, "end_time":1000},]
        :param asr_results: json_string
        :return: {
        "segment":[
        {"begin_time":0, "end_time":1000, "score":0.8911, "label":1},
        {"begin_time":2000, "end_time":3000, "score":0.2321, "label":0},
        {"begin_time":5000, "end_time":8000, "score":0.7872, "label":1}
        ],
        "total": {"score": 0.6, "label": 1}
        }
        '''

        default_result = {
            "segment": [],
            "total": {
                "score": 0.0,
                "label": 0
            }
        }

        temp_dir = None

        try:

            if wav_clips_list is None or asr_results is None:
                print_error("ERROR: some required positional arguments are missed.")
                return json.dumps({"code": 0, "msg": "failed", "data": default_result})

            print("开始预测，收到输入的音频切片数量为:{}".format(len(wav_clips_list)))

            if len(wav_clips_list) == 0:
                # print("输入列表为空，返回默认值。")
                return json.dumps({"code": 0, "msg": "failed", "data": default_result})

            uid = uuid.uuid1()
            temp_dir = os.path.join(base_path, "temp_{}".format(uid))
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
            # print("创建文件夹：{}".format(temp_dir))

            # asr_results = json.loads(asr_result_str)["data"]["result"]
            asr_clips_list = []

            interval_times = {}
            for one in wav_clips_list:
                clip_path = one["wav_path"]
                clip_name = clip_path.split("/")[-1]

                begin, end = one["begin_time"], one["end_time"]

                interval_times[clip_name] = (begin, end)

                input_dict = {
                    "result": [],
                }

                in_flag = False

                for line in asr_results:
                    asr_begin = int(line["begin_time"])
                    asr_end = int(line["end_time"])

                    if asr_begin == begin:
                        in_flag = True

                    asr_text = line["text"]
                    asr_sentence_id = int(line["sentence_id"])

                    if in_flag:
                        input_dict["result"].append({
                            "begin_time": asr_begin,
                            "end_time": asr_end,
                            "sentence_id": asr_sentence_id,
                            "status_code": 0,
                            "text": asr_text,
                        })

                    if asr_end == end:
                        in_flag = False

                asr_clips_list.append(input_dict)

            wav_clips = {}
            asr_clips = {}
            for i in range(len(wav_clips_list)):
                clip_path = wav_clips_list[i]["wav_path"]
                asr_res = asr_clips_list[i]
                clip_name = clip_path.split("/")[-1]
                wav_clips[clip_name] = clip_path
                asr_clips[clip_name] = asr_res

            clips_op_features = opensmile_extraction_fluency.batch_extract(wav_clips, temp_dir)
            clips_col_features = extract_columbus_features.batch_extract(asr_clips, wav_clips)

            result = {
                "segment": [],
                "total": {}
            }
            scores = []

            for clip in clips_op_features:
                try:
                    features = np.concatenate([clips_col_features[clip], clips_op_features[clip]], axis=1)

                    embdding = self.inference(features)

                    label = int(self.lr_model.predict(embdding)[0])
                    score = float(self.lr_model.predict_proba(embdding)[0][-1])

                    begin_time, end_time = interval_times[clip]

                    result["segment"].append(
                        {
                            "begin_time": begin_time,
                            "end_time": end_time,
                            "score": score,
                            "label": label
                        }
                    )
                    scores.append(score)
                except Exception as e:
                    print_error("ERROR: predict clip {} crash, the error is {}".format(clip, str(e)))
                    result["segment"].append(
                        {"begin_time": 0,
                         "end_time": 0,
                         "score": 1.0,
                         "label": 1,
                         }
                    )

            total_score = float(np.mean(scores))
            if total_score > 0.5:
                total_label = 1
            else:
                total_label = 0

            result["total"]["score"] = total_score
            result["total"]["label"] = total_label

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("删除文件夹：{}".format(temp_dir))

            return result

        except Exception as e:
            print_error(str(e))
            raise e

        finally:
            if temp_dir is not None:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print("删除文件夹：{}".format(temp_dir))


if __name__=="__main__":

    audio_fluency_model = Fluency(rll_model_path=os.path.join(base_path,"RLL_l1_128_l2_128_lr_0.05_penalty_5.0_bs_512_dropout_5.0_hard_2_sim_euc_anchor_mass_anchor_ON_kangyu_beike_e_online"),lr_model_path=os.path.join(base_path, "lr.pkl"))
    wav_list =[{"wav_path":"./test.wav", "begin_time":0, "end_time":1000},]
    asr_result= [{"end_time": "1330", "begin_time": "630", "text": "大家的小。", "sentence_id": 1}]
    

    result = audio_fluency_model.predict(wav_list,asr_result)
    print(result)
