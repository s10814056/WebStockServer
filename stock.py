from flask import Flask, render_template, request, jsonify, json
import requests
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import os
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/stock/<string:code>/<string:name>", methods=['GET'])
def stock_code(code,name):
    
    stock_name=name
    stock_code=code
    
    output_dir = './model_save1/'

    #os.chdir(r'C:\Users\sleep\Desktop')

    model =TFBertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    #主程式主程式主程式主程式主程式主程式主程式主程式主程式主程式主程式主程式主程式主程式
     # 要抓取的網址
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
    url="https://tw.stock.yahoo.com/quote/"+stock_code+"/news"
    #print(url)

    #請求網站
    list_req = requests.get(url, headers=headers)
      #將整個網站的程式碼爬下來
    soup = BeautifulSoup(list_req.content, "html.parser")
      #找到b這個標籤

    now =time.localtime()  #當日時間
    now = str(now.tm_year)+'年'+str(now.tm_mon)+'月'+ str(now.tm_mday)+'日' 
    num=now.find('日')

    b=0
    newData={}
    all=[]

    for a in soup.find_all('h3',{'class':'Mt(0) Mb(8px)'}):
        if stock_name in a.text:
            url = a.a.get('href')
            list_req = requests.get(url)
            soup1 = BeautifulSoup(list_req.content, "html.parser")
            getAllNew= soup1.find('div',{'class':'caas-body'}) 
            gettime= soup1.find('time',{'class':'caas-attr-meta-time'}) #抓日期
            #標題
            newData['title']=a.text
            newData['date']=gettime.get_text('datetime')[0:10]
            newData['content']=getAllNew.text
            newData['url']=url
            all.append(newData)
            newData={}
            b+=1
            if b>=5:
                break
    for f in range(5):
        pre_text= [all[f]['content']]
        tf_batch = tokenizer(pre_text[0][11:], max_length=128, padding=True, truncation=True, return_tensors='tf')
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels = ['非常好','好','普通','不好','非常不好']   #5最好 1最差
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        all[f]['score']=labels[label[0]]
    all=tuple(all)
    return jsonify(all)

@app.route("/news/<path:url>", methods=['GET'])
def news_url(url):
    stock_new_url = url
    output_dir = './model_save1/'

    #os.chdir(r'C:\Users\sleep\Desktop')

    model =TFBertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    
    response = requests.get(stock_new_url)
    soup = BeautifulSoup(response.text, "html.parser")         
    
    newData=[]
    h1=soup.find('h1')
    content= soup.find('div',{'class':'caas-body'}) 
    time=soup.find('time',{'class':'caas-attr-meta-time'})
    newData.append([h1.text,time.text,content.text])



    pre_text= [newData[0][2]]
    tf_batch = tokenizer(pre_text[0][11:], max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['非常好','好','普通','不好','非常不好']  #好到壞
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    newData[0].append(labels[label[0]])
    newData=tuple(newData)
    return jsonify(newData) 

if __name__ == '__main__':
    app.run()