from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import plotly
import plotly.graph_objects as go
import chart_studio.plotly as csp
import json

# df = pd.read_excel(
# 'dfgab.xlsx')

# # print(df)

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')#, tables=[df.to_html(classes='data')], titles=df.columns.values)


@app.route('/convertedtousd')
def dfgab():
    return render_template('dfgab.html')

@app.route('/allratio')
def dfallratio():
    return render_template('concatall.html')

@app.route('/uncoverted')
def dfunconverted():
    return render_template('unconcal.html')

@app.route('/kurs')
def dfkurs():
    return render_template('kurscal.html')

@app.route('/converted')
def dfconverted():
    return render_template('converted.html')

###
@app.route('/predictindo', methods = ['POST', 'GET'])
def predictindo():
    # if request.method == 'POST':
    input = request.form
    kindo = (input['kursindo: ']).split(',')
    kuindo = []
    for i in kindo:
        a  = float(i)
        kuindo.append(a)
    kuindo = np.array(kuindo)
    if kuindo.shape[0] == 1:
        kuindo = kuindo.reshape(1,1)
    else:
        kuindo = np.array(kuindo).reshape(-1,1)
    kursindo = modelindo.predict(kuindo)

    
    x = (input['insertindo: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kursindo):
        divide = q / kursindo
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrindo = modelindoi.predict(divide)
            ylrindo
            ylrconindo = ylrindo * kursindo
            ylrconindo
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrindo = modelindoi.predict(divide)
            ylrindo
            ylrconindo = ylrindo * kursindo
            ylrconindo
    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrindo = divide
        ylrconindo = divide
    return render_template('predictindo.html', data=input, predindo=ylrconindo, predkindo=kursindo, predgab=divide) ##immutabledit
####


@app.route('/predictlaos', methods = ['POST', 'GET'])
def predict():
    # if request.method == 'POST':
    input = request.form
    klaos = (input['kurs: ']).split(',')
    kulaos = []
    for i in klaos:
        a  = float(i)
        kulaos.append(a)
    kulaos = np.array(kulaos)
    if kulaos.shape[0] == 1:
        kulaos = kulaos.reshape(1,1)
    else:
        kulaos = np.array(kulaos).reshape(-1,1)
    kurslaos = modellaos.predict(kulaos)

    
    x = (input['insert: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kurslaos):
        divide = q / kurslaos
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrlaos = modellaosi.predict(divide)
            ylrlaos
            ylrconlaos = ylrlaos * kurslaos
            ylrconlaos
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrlaos = modellaosi.predict(divide)
            ylrlaos
            ylrconlaos = ylrlaos * kurslaos
            ylrconlaos

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrlaos = divide
        ylrconlaos = divide
    return render_template('predictlaos.html', data=input, predlaos=ylrconlaos, predklaos=kurslaos, predgab=divide)

@app.route('/predictmalay', methods = ['POST', 'GET'])
def predictmalay():
    # if request.method == 'POST':
    input = request.form
    kmalay = (input['kursmalay: ']).split(',')
    kumalay = []
    for i in kmalay:
        a  = float(i)
        kumalay.append(a)
    kumalay = np.array(kumalay)
    if kumalay.shape[0] == 1:
        kumalay = kumalay.reshape(1,1)
    else:
        kumalay = np.array(kumalay).reshape(-1,1)
    kursmalay = modelmalay.predict(kumalay)

    
    x = (input['insertmalay: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kursmalay):
        divide = q / kursmalay
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrmalay = modelmalayi.predict(divide)
            ylrmalay
            ylrconmalay = ylrmalay * kursmalay
            ylrconmalay
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrmalay = modelmalayi.predict(divide)
            ylrmalay
            ylrconmalay = ylrmalay * kursmalay
            ylrconmalay

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrmalay = divide
        ylrconmalay = divide
    return render_template('predictmalay.html', data=input, predmalay=ylrconmalay, predkmalay=kursmalay, predgab=divide)

@app.route('/predictphilip', methods = ['POST', 'GET'])
def predictphilip():
    # if request.method == 'POST':
    input = request.form
    kphilip = (input['kursphilip: ']).split(',')
    kuphilip = []
    for i in kphilip:
        a  = float(i)
        kuphilip.append(a)
    kuphilip = np.array(kuphilip)
    if kuphilip.shape[0] == 1:
        kuphilip = kuphilip.reshape(1,1)
    else:
        kuphilip= np.array(kuphilip).reshape(-1,1)
    kursphilip = modelmalay.predict(kuphilip)

    
    x = (input['insertphilip: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kursphilip):
        divide = q / kursphilip
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrphilip = modelphilipi.predict(divide)
            ylrphilip
            ylrconphilip = ylrphilip * kursphilip
            ylrconphilip
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrphilip = modelphilipi.predict(divide)
            ylrphilip
            ylrconphilip = ylrphilip * kursphilip
            ylrconphilip

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrphilip = divide
        ylrconphilip = divide
    return render_template('predictphilip.html', data=input, predphilip=ylrconphilip, predkphilip=kursphilip, predgab=divide)

@app.route('/predictsing', methods = ['POST', 'GET'])
def predictsing():
    # if request.method == 'POST':
    input = request.form
    ksing = (input['kurssing: ']).split(',')
    kusing = []
    for i in ksing:
        a  = float(i)
        kusing.append(a)
    kusing = np.array(kusing)
    if kusing.shape[0] == 1:
        kusing = kusing.reshape(1,1)
    else:
        kusing= np.array(kusing).reshape(-1,1)
    kurssing= modelmalay.predict(kusing)

    
    x = (input['insertsing: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kurssing):
        divide = q / kurssing
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrsing = modelsingi.predict(divide)
            ylrsing
            ylrconsing = ylrsing * kurssing
            ylrconsing
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrsing = modelsingi.predict(divide)
            ylrsing
            ylrconsing = ylrsing * kurssing
            ylrconsing

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrsing = divide
        ylrconsing = divide
    return render_template('predictsing.html', data=input, predsing=ylrconsing, predksing=kurssing, predgab=divide)


@app.route('/predictthai', methods = ['POST', 'GET'])
def predictthai():
    # if request.method == 'POST':
    input = request.form
    kthai = (input['kursthai: ']).split(',')
    kuthai = []
    for i in kthai:
        a  = float(i)
        kuthai.append(a)
    kuthai = np.array(kuthai)
    if kuthai.shape[0] == 1:
        kuthai = kuthai.reshape(1,1)
    else:
        kuthai= np.array(kuthai).reshape(-1,1)
    kursthai= modelthai.predict(kuthai)

    
    x = (input['insertthai: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kursthai):
        divide = q / kursthai
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrthai = modelthaii.predict(divide)
            ylrthai
            ylrconthai = ylrthai * kursthai
            ylrconthai
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrthai = modelthaii.predict(divide)
            ylrthai
            ylrconthai = ylrthai * kursthai
            ylrconthai

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrthai = divide
        ylrconthai = divide
    return render_template('predictthai.html', data=input, predthai=ylrconthai, predkthai=kursthai, predgab=divide)

@app.route('/predictviet', methods = ['POST', 'GET'])
def predictviet():
    # if request.method == 'POST':
    input = request.form
    kviet = (input['kursviet: ']).split(',')
    kuviet = []
    for i in kviet:
        a  = float(i)
        kuviet.append(a)
    kuviet = np.array(kuviet)
    if kuviet.shape[0] == 1:
        kuviet = kuviet.reshape(1,1)
    else:
        kuviet= np.array(kuviet).reshape(-1,1)
    kursviet= modelviet.predict(kuviet)

    
    x = (input['insertviet: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)

    if len(q) == len(kursviet):
        divide = q / kursviet
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrviet = modelvieti.predict(divide)
            ylrviet
            ylrconviet = ylrviet * kursviet
            ylrconviet
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrviet = modelvieti.predict(divide)
            ylrviet
            ylrconviet = ylrviet * kursviet
            ylrconviet

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrviet = divide
        ylrconviet = divide
    return render_template('predictviet.html', data=input, predviet=ylrconviet, predkviet=kursviet, predgab=divide)



@app.route('/peta')
def peta():
    return render_template('petasharpe.html')

@app.route('/sharpe')
def sharpe():
    return render_template('sharpe.html')

@app.route('/plot')
def plot():
    df = pd.read_csv('sharperatio1.csv')
    ylist = df['sharpe ratio'].tolist()
    xlist = df['countries']
    plot = go.Bar(
        x = xlist,
        y = ylist
    )
    graphJSON = json.dumps([plot], cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plot.html', plot=graphJSON)

# @app.route('/storage/<path:x>')
# def staticfile(x):
#     return send_from_directory('myfile', x)


if __name__== '__main__':
    modelindoi = joblib.load('regressionindoi')
    modelindo = joblib.load('regressionindo')
    modellaosi = joblib.load('regressionlaosi')
    modellaos = joblib.load('regressionlaos')
    modelmalayi = joblib.load('regressionmalayi')
    modelmalay= joblib.load('regressionmalay')
    modelphilipi = joblib.load('regressionphilipi')
    modelphilip= joblib.load('regressionphilip')
    modelsingi = joblib.load('regressionsingi')
    modelsing= joblib.load('regressionsing')
    modelthaii = joblib.load('regressionthaii')
    modelthai= joblib.load('regressionthai')
    modelvieti = joblib.load('regressionvieti')
    modelviet= joblib.load('regressionviet')
    app.run(debug=True)
