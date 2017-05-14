# Flask microservice of text -> tag

from __future__ import print_function

from flask import Flask, Request
from flask import request
app = Flask(__name__)

import json
import pickle

from gensim import corpora, models, utils
from sklearn.externals import joblib
import numpy as np


class TextTagger(object):
  """Object which tags articles. Needs topic modeler and """
  def __init__(self, topic_modeler, gensim_dict, lr_dict, threshold=0.5):
    super(TextTagger, self).__init__()
    self.topic_modeler = topic_modeler
    self.gensim_dict = gensim_dict
    self.lr_dict = lr_dict
    self.threshold = threshold

  def text_to_topic_list(self, text):
    text = text.lower()
    tokens = list(utils.tokenize(text))
    bow = self.gensim_dict.doc2bow(tokens)
    return self.topic_modeler[bow]    

  def text_to_numpy(self, text):
    out = np.zeros(self.topic_modeler.num_topics)
    for idx, val in self.text_to_topic_list(text):
      out[idx] = val
    return out
    
  def text_to_topic_dict(self, text):
    return {topic: weight for topic, weight in self.label_article(text)}

  def text_to_tags(self, text, debug=False):
    input_vect = np.array([self.text_to_numpy(text)])
    tags = []
    for label, lr_model in self.lr_dict.items():
      tag_prob = lr_model.predict_proba(input_vect)[0, 1]
      if debug:
        print(label, tag_prob)
      if tag_prob > self.threshold:
        tags.append(label)

    tags = ["Programming" if t == "Javascript" else t for t in tags]
    return tags

  @classmethod
  def init_from_files(cls, topic_model_fname, gensim_dict_fname, lr_dict_fname,
                      *args, **kwargs):
    topic_modeler = models.ldamodel.LdaModel.load(topic_model_fname)
    gensim_dict = corpora.Dictionary.load(gensim_dict_fname)
    lr_dict = joblib.load(lr_dict_fname)
    return cls(topic_modeler, gensim_dict, lr_dict, *args, **kwargs)
    

text_tagger = TextTagger.init_from_files(
  "models/model_100topics_10passMay14_0259.gensim", 
  "models/hn_dictionaryMay14_0240.pkl", 
  "models/serialized_model/rf_models.pkl", 
  threshold=0.3,
)


@app.route("/tag", methods=['POST'])
def tag_text():
  data_obj = request.get_json()
  text = data_obj.get("text", "")
  tag_list = text_tagger.text_to_tags(text)
  output = {"tags": tag_list}
  return json.dumps(output)


@app.route('/')
def hello_world():
  return "hello world"
