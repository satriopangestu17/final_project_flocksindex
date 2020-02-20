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
    kuindo1 = []
    for i in kindo:
        a  = float(i)
        kuindo1.append(a)
    kuindo1 = np.array(kuindo1)
    if kuindo1.shape[0] == 1:
        kuindo = kuindo1.reshape(1,1)
    else:
        kuindo = np.array(kuindo1).reshape(-1,1)
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
        divide = q / kuindo1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrindo = modelindoi.predict(divide)
            ylrindo
            ylrconindo = ylrindo * kursindo
            ylrconindo
            conawal = q/kuindo1
            returncona = []
            for i in range(len(ylrindo)):
                returncon = (ylrindo[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrindo = modelindoi.predict(divide)
            ylrindo
            ylrconindo = ylrindo * kursindo
            ylrconindo
            conawal = q/kuindo1
            returncona = []
            for i in range(len(ylrindo)):
                returncon = (ylrindo[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrindo = divide
        ylrconindo = divide
        conawal = divide
        returncona = divide
        
    return render_template('predictindo.html', data=input, predindo=ylrconindo, predkindo=kursindo, predgab=ylrindo, conawal=conawal, returncona=returncona) ##immutabledit
####


@app.route('/predictlaos', methods = ['POST', 'GET'])
def predict():
    # if request.method == 'POST':
    input = request.form
    klaos = (input['kurs: ']).split(',')
    kulaos1 = []
    for i in klaos:
        a  = float(i)
        kulaos1.append(a)
    kulaos1 = np.array(kulaos1)
    if kulaos1.shape[0] == 1:
        kulaos = kulaos1.reshape(1,1)
    else:
        kulaos = np.array(kulaos1).reshape(-1,1)
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
        divide = q / kulaos1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrlaos = modellaosi.predict(divide)
            ylrlaos
            ylrconlaos = ylrlaos * kurslaos
            ylrconlaos
            conawal = q/kulaos1
            returncona = []
            for i in range(len(ylrlaos)):
                returncon = (ylrlaos[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrlaos = modellaosi.predict(divide)
            ylrlaos
            ylrconlaos = ylrlaos * kurslaos
            ylrconlaos
            conawal = q/kulaos1
            returncona = []
            for i in range(len(ylrlaos)):
                returncon = (ylrlaos[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrlaos = divide
        ylrconlaos = divide
        conawal = divide
        returncona = divide
        
    return render_template('predictlaos.html', data=input, predlaos=ylrconlaos, predklaos=kurslaos, predgab=ylrlaos, conawal=conawal, returncona=returncona)

@app.route('/predictmalay', methods = ['POST', 'GET'])
def predictmalay():
    # if request.method == 'POST':
    input = request.form
    kmalay = (input['kursmalay: ']).split(',')
    kumalay1 = []
    for i in kmalay:
        a  = float(i)
        kumalay1.append(a)
    kumalay1 = np.array(kumalay1)
    if kumalay1.shape[0] == 1:
        kumalay = kumalay1.reshape(1,1)
    else:
        kumalay = np.array(kumalay1).reshape(-1,1)
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
        divide = q / kumalay1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrmalay = modelmalayi.predict(divide)
            ylrmalay
            ylrconmalay = ylrmalay * kursmalay
            ylrconmalay
            conawal = q/kumalay1
            returncona = []
            for i in range(len(ylrmalay)):
                returncon = (ylrmalay[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrmalay = modelmalayi.predict(divide)
            ylrmalay
            ylrconmalay = ylrmalay * kursmalay
            ylrconmalay
            conawal = q/kumalay1
            returncona = []
            for i in range(len(ylrmalay)):
                returncon = (ylrmalay[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrmalay = divide
        ylrconmalay = divide
        conawal = divide
        returncona = divide
        
    return render_template('predictmalay.html', data=input, predmalay=ylrconmalay, predkmalay=kursmalay, predgab=ylrmalay, conawal=conawal, returncona=returncona)

@app.route('/predictphilip', methods = ['POST', 'GET'])
def predictphilip():
    # if request.method == 'POST':
    input = request.form
    kphilip = (input['kursphilip: ']).split(',')
    kuphilip1 = []
    for i in kphilip:
        a  = float(i)
        kuphilip1.append(a)
    kuphilip1 = np.array(kuphilip1)
    if kuphilip1.shape[0] == 1:
        kuphilip = kuphilip1.reshape(1,1)
    else:
        kuphilip= np.array(kuphilip1).reshape(-1,1)
    kursphilip = modelmalay.predict(kuphilip)

    
    x = (input['insertphilip: ']).split(',')
# x = np.array(x)
    q = []
    for i in range(len(x)):
        c = float(x[i])
    #     d = c / kurslaos[i]
        q.append(c)
    q = np.array(q)
    conawal =  q/kuphilip1

    if len(q) == len(kursphilip):
        divide = q / kuphilip1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrphilip = modelphilipi.predict(divide)
            ylrphilip
            ylrconphilip = ylrphilip * kursphilip
            ylrconphilip
            conawal = q /kuphilip1
            returncona = []
            for i in range(len(ylrphilip)):
                returncon = (ylrphilip[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrphilip = modelphilipi.predict(divide)
            ylrphilip
            ylrconphilip = ylrphilip * kursphilip
            ylrconphilip
            conawal = q/kuphilip1
            returncona = []
            for i in range(len(ylrphilip)):
                returncon = (ylrphilip[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrphilip = divide
        ylrconphilip = divide
        conawal = divide
        returncona = divide
        
    return render_template('predictphilip.html', data=input, predphilip=ylrconphilip, predkphilip=kursphilip, predgab=ylrphilip, conawal=conawal, returncona=returncona)

@app.route('/predictsing', methods = ['POST', 'GET'])
def predictsing():
    # if request.method == 'POST':
    input = request.form
    ksing = (input['kurssing: ']).split(',')
    kusing1 = []
    for i in ksing:
        a  = float(i)
        kusing1.append(a)
    kusing1 = np.array(kusing1)
    if kusing1.shape[0] == 1:
        kusing = kusing1.reshape(1,1)
    else:
        kusing= np.array(kusing1).reshape(-1,1)
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
        divide = q / kusing1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrsing = modelsingi.predict(divide)
            ylrsing
            ylrconsing = ylrsing * kurssing
            ylrconsing
            conawal = q/kusing1
            returncona = []
            for i in range(len(ylrsing)):
                returncon = (ylrsing[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrsing = modelsingi.predict(divide)
            ylrsing
            ylrconsing = ylrsing * kurssing
            ylrconsing
            conawal = q/kusing1
            returncona = []
            for i in range(len(ylrsing)):
                returncon = (ylrsing[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrsing = divide
        ylrconsing = divide
        conawal = divide
        returncona = divide
    return render_template('predictsing.html', data=input, predsing=ylrconsing, predksing=kurssing, predgab=ylrsing, conawal=conawal, returncona=returncona)


@app.route('/predictthai', methods = ['POST', 'GET'])
def predictthai():
    # if request.method == 'POST':
    input = request.form
    kthai = (input['kursthai: ']).split(',')
    kuthai1 = []
    for i in kthai:
        a  = float(i)
        kuthai1.append(a)
    kuthai1 = np.array(kuthai1)
    if kuthai1.shape[0] == 1:
        kuthai = kuthai1.reshape(1,1)
    else:
        kuthai= np.array(kuthai1).reshape(-1,1)
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
        divide = q / kuthai1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrthai = modelthaii.predict(divide)
            ylrthai
            ylrconthai = ylrthai * kursthai
            ylrconthai
            conawal = q/kuthai1
            returncona = []
            for i in range(len(ylrthai)):
                returncon = (ylrthai[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrthai = modelthaii.predict(divide)
            ylrthai
            ylrconthai = ylrthai * kursthai
            ylrconthai
            conawal = q/kuthai1
            returncona = []
            for i in range(len(ylrthai)):
                returncon = (ylrthai[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrthai = divide
        ylrconthai = divide
        conawal = divide
        returncona  = divide
    return render_template('predictthai.html', data=input, predthai=ylrconthai, predkthai=kursthai, predgab=ylrthai, conawal=conawal, returncona=returncona)

@app.route('/predictviet', methods = ['POST', 'GET'])
def predictviet():
    # if request.method == 'POST':
    input = request.form
    kviet = (input['kursviet: ']).split(',')
    kuviet1 = []
    for i in kviet:
        a  = float(i)
        kuviet1.append(a)
    kuviet1 = np.array(kuviet1)
    if kuviet1.shape[0] == 1:
        kuviet = kuviet1.reshape(1,1)
    else:
        kuviet= np.array(kuviet1).reshape(-1,1)
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
        divide = q / kuviet1
        print(divide)
        if divide.shape[0] == 1:
            divide = divide.reshape(1,1)
            ylrviet = modelvieti.predict(divide)
            ylrviet
            ylrconviet = ylrviet * kursviet
            ylrconviet
            conawal = q/kuviet1
            returncona = []
            for i in range(len(ylrviet)):
                returncon = (ylrviet[i]-conawal[i])/conawal[i]
                returncona.append(returncon)
        else:
            divide = np.array(divide).reshape(-1,1)
            ylrviet = modelvieti.predict(divide)
            ylrviet
            ylrconviet = ylrviet * kursviet
            ylrconviet
            conawal =q/kuviet1
            returncona = []
            for i in range(len(ylrviet)):
                returncon = (ylrviet[i]-conawal[i])/conawal[i]
                returncona.append(returncon)

    else:
    #     print('length should be the same')
        divide = 'put in your currency and index in the same length!'
        ylrviet = divide
        ylrconviet = divide
        conawal = divide
        returncona = divide
    return render_template('predictviet.html', data=input, predviet=ylrconviet, predkviet=kursviet, predgab=ylrviet, conawal=conawal, returncona=returncona)



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
