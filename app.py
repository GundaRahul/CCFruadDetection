from flask import Flask, render_template, request
import flask
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, multivariate_normal

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

def make_classification_prediction(X):
	loaded_model = pickle.load(open("CC_Fraud.pkl", 'rb'))
	result = loaded_model.predict(X)
	return result

def MultivariateGaussianDistribution(data,mu,sigma):
    p = multivariate_normal.pdf(data, mean=mu, cov=sigma)
    p_transformed = np.power(p,1/100) #transformed the probability scores by p^1/100 since the values are very low (up to e-150)
    return p_transformed

def MeanAndSigmaValues():
	mu=pd.read_csv('./Data/mu.csv',squeeze=True,header=None,index_col=0)
	sigma=np.load( './Data/sigma.npy' )
	return mu, sigma

if __name__ == '__main__':
	app.run(debug=True)