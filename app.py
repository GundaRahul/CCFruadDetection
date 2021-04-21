from flask import Flask, render_template, request
import flask
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, multivariate_normal
from keras.models import load_model

app = Flask(__name__)

actual_low=[]
actual_high=[]
pred_low=[]
pred_high=[]

@app.route('/a', methods=['GET','POST'])
def index():
	return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def classification():
	if request.method=="POST":
		return "Welcome to DataJango Classification Page"
	elif request.method=="GET":
		data = pd.read_csv("../Data/datafile_CreditCard.csv")
		data = data.sample(frac=1).reset_index(drop=True)
		data1=data.iloc[:3,:]
		data2=data[data['Class']==1][:2]
		data_final=pd.concat([data1, data2], ignore_index=True)
		data_final = data_final.sample(frac=1).reset_index(drop=True)
		data_y = data_final['Class']
		data_final.drop(labels='Class',axis=1,inplace=True)
		mu, sigma = MeanAndSigmaValues()
		p_temp = MultivariateGaussianDistribution(data_final,mu,sigma)
		predicted_values = make_classification_prediction(data_final)
		data_final1 = (p_temp < 0.4)
		data_final = data_final.round(3)

		return render_template('Classification.html', data=data_final,zip=zip,actual_data=data_y,data_predict1=data_final1,data_predict2=predicted_values,len=len)

def template(text = ""):
    templateDate = {
        'text' : text,
        'tvalues' : getTValues(),
        'selected_tvalue' : "HDFC Bank"
    }
    return templateDate

#Add Stock holder/Company names as required ex: "ICICI Bank", "Andra Bank"
def getTValues():
    return ("HDFC Bank") 

@app.route("/stockMarket/", methods=['POST', 'GET'])
def stockMarket():
    tvalue = "HDFC Bank" #default value
    msg = 'HDFC Bank Prediction'
    if request.method == "POST":            
        tvalue = request.form['tvalue']
        print("------------------" + tvalue)
        msg= tvalue + " Prediction"
    templateData = template(text = msg)
    templateData['selected_tvalue'] = tvalue 

    datafile= pd.read_csv('../Data/datafile_HDFC.csv',usecols=['Date'])
    data1 = pd.read_csv('../Data/RNN_data.csv')
    low_model = load_model('../PickelFile/HDFC_low_model__.h5')
    high_model = load_model('../PickelFile/HDFC_high_model__.h5')
    pred_low,pred_high = predict_stock_price(data1.iloc[-20:],low_model,high_model)
    data_display = data1[["High Price","Low Price"]].iloc[-10:]
    PrevPredValues=StockMarketPredictedValues(data1,datafile,10,low_model,high_model)
    ProfitLoss = []
    #ActLow, PredLow, PredHigh, ActHigh
    for AL,PL,PH,AH in zip(data_display['Low Price'], PrevPredValues['pred_low'], PrevPredValues['pred_high'], data_display['High Price']):
        ProfitLoss.append(ProfitLossPrediction(AL,PL,PH,AH))

    return render_template('StockMarket.html', **templateData, ProfitLoss=ProfitLoss, PrevPredValues=PrevPredValues, zip=zip, data=data_display, predLow=pred_low, predHigh = pred_high)

def make_classification_prediction(X):
	loaded_model = pickle.load(open("../PickelFile/CC_Fraud.pkl", 'rb'))
	result = loaded_model.predict(X)
	return result

def make_wp_classification_prediction(X):
	loaded_model = pickle.load(open("../PickelFile/waterplant.pkl", 'rb'))
	result = loaded_model.predict(X)
	return result

def predict_low_price(x_20_data,low_model):
    if(x_20_data.shape[0] == 20):
        input_20 = x_20_data.reshape((-1,20,8))
        return low_model.predict(input_20)

def predict_high_price(x_20_data,high_model):
    if(x_20_data.shape[0] == 20):
        input_20 = x_20_data.reshape((-1,20,8))
        return high_model.predict(input_20)

def predict_stock_price(X,low_model,high_model):
    X_low = X[['Open Price','High Price', 'Last Price', 'Average Price', \
                'Total Traded Quantity', 'Turnover', 'Prev Close','Close Price']].values
               
    y_low = X['Low Price'].values
    X_high = X[['Open Price','Low Price', 'Last Price', 'Average Price', \
                'Total Traded Quantity', 'Turnover', 'Prev Close','Close Price']].values
    y_high = X['High Price'].values
    scaling_low= pickle.load(open('../PickelFile/scaling_low.pkl', 'rb'))
    scaling_high= pickle.load(open('../PickelFile/scaling_high.pkl', 'rb'))
    X_low = scaling_low.transform(X_low)
    X_high = scaling_high.transform(X_high)
    return predict_low_price(X_low,low_model), predict_high_price(X_high,high_model)

def MultivariateGaussianDistribution(data,mu,sigma):
    p = multivariate_normal.pdf(data, mean=mu, cov=sigma)
    p_transformed = np.power(p,1/100) #transformed the probability scores by p^1/100 since the values are very low (up to e-150)
    return p_transformed

def MeanAndSigmaValues():
	mu=pd.read_csv('../Data/mu.csv',squeeze=True,header=None,index_col=0)
	sigma=np.load( '../Data/sigma.npy' )
	return mu, sigma

def StockMarketPredictedValues(data,dateCol,number,low_model,high_model):
    X_low = data[['Open Price','High Price' , 'Last Price', 'Average Price', 'Close Price', \
            'Total Traded Quantity', 'Turnover',  'Prev Close']].values
    y_low = data['Low Price'].values
    X_high = data[['Open Price','Low Price' , 'Last Price', 'Average Price', \
                'Total Traded Quantity', 'Turnover',  'Prev Close','Close Price']].values
    y_high = data['High Price'].values
    date=dateCol[['Date']].values
    scaling_low= pickle.load(open('../PickelFile/scaling_low.pkl', 'rb'))
    scaling_high= pickle.load(open('../PickelFile/scaling_high.pkl', 'rb'))
    X_low1 = scaling_low.transform(X_low)
    X_high1 = scaling_high.transform(X_high)
    final=pd.DataFrame(columns=['Date','actual_low','pred_low','actual_high','pred_high'])

    for j in range(-number-1,-1):
        i=j-20
        low=X_low1[i:j]
        high=X_high1[i:j]
        actual_low.append( y_low[j+1] )
        pred_low.append( round(predict_low_price(low,low_model)[-1].item(),2) )
        actual_high.append( y_high[j+1] )
        pred_high.append( round(predict_high_price(high,high_model)[-1].item(),2) )
        final = final.append({'Date':date[j+1][-1],'actual_low':actual_low[-1], \
                              'pred_low':pred_low[-1],'actual_high':actual_high[-1],\
                              'pred_high':pred_high[-1]}, ignore_index=True)

    return final

def ProfitLossPrediction(ActLow, PredLow, PredHigh, ActHigh):
    #for AL,PL,PH,AH in zip(ActLow, PredLow, PredHigh, ActHigh):
    if ActLow <= PredLow and PredHigh <= ActHigh :
        profit = ((PredHigh-PredLow)/PredLow)*100
        return round(profit,2)
    else:
        return "No Profit"
        #loss = ((PredHigh-PredLow)/PredLow)*100
        #return loss

if __name__ == '__main__':
	app.run(debug=True)