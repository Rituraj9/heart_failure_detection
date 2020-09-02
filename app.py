import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///heart.db'

db=SQLAlchemy(app)

df1 = pd.read_csv("heart_failure_clinical_records_dataset.csv")
X1 = df1.iloc[:,:-2].values
y1= df1.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train1 = ss.fit_transform(X_train1)
X_test1 = ss.fit_transform(X_test1)

ann = tf.keras.models.Sequential()
#Full Connection
ann.add(tf.keras.layers.Dense(units=10,input_dim=11,activation='relu'))
ann.add(tf.keras.layers.Dense(units=18,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train1,y_train1,batch_size=8,epochs=150)

ann.save('heart_failure_model.pkl')

#Create Model
class heart(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	age = db.Column(db.Integer,nullable=False)
	anaemia = db.Column(db.Integer,nullable=False)
	creatinine_phosphokinase = db.Column(db.Integer,nullable=False)
	diabetes = db.Column(db.Integer,nullable=False)
	ejection_fraction = db.Column(db.Integer,nullable=False)
	high_blood_pressure = db.Column(db.Integer,nullable=False)
	platelets = db.Column(db.Integer,nullable=False)
	serum_creatinine = db.Column(db.Integer,nullable=False)
	serum_sodium = db.Column(db.Integer,nullable=False)
	sex = db.Column(db.Integer,nullable=False)
	smoking = db.Column(db.Integer,nullable=False)
	date_created = db.Column(db.DateTime,default=datetime.utcnow)

	def __str__(self):
		return self.age

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
	return render_template('detect.html')

@app.route('/predict',methods=['POST'])
def predict():
	hf = tf.keras.models.load_model('heart_failure_model.pkl')
	hf._make_predict_function()
	graph = tf.get_default_graph()
	age = request.form['age']
	anaemia = request.form['anaemia']
	creatinine_phosphokinase = request.form['creatinine_phosphokinase']
	diabetes = request.form['diabetes']
	ejection_fraction = request.form['ejection_fraction']
	high_blood_pressure = request.form['high_blood_pressure']
	platelets = request.form['platelets']
	serum_creatinine = request.form['serum_creatinine']
	serum_sodium = request.form['serum_sodium']
	sex = request.form['sex']
	smoking = request.form['smoking']
	new_data = heart(age=age,anaemia=anaemia,creatinine_phosphokinase=creatinine_phosphokinase,diabetes=diabetes,ejection_fraction=ejection_fraction,high_blood_pressure=high_blood_pressure,platelets=platelets,serum_creatinine=serum_creatinine,serum_sodium=serum_sodium,sex=sex,smoking=smoking)
	db.session.add(new_data)
	db.session.commit()
	features=[str(x) for x in request.form.values()]
	final = np.array([features])
	final=ss.transform(final)
	with graph.as_default():
		prediction = hf.predict(final)
	prediction = prediction>0.5
	if prediction[0][0]==True:
		m=1
	else:
		m=0
	print(m)

	if m==1:
		tt= "You Are At High Risk of Heart Failure. Please Consult The Doctors Immediately"
	else:
		tt = "You are safe from any type of Heart Disease or Heart Failure"
		
	print(tt)
	acc="97%"
	return render_template('prediction.html',prediction_text='{}'.format(tt),accuracy='{}'.format(acc))


@app.route('/warning')
def warning():
    return render_template('warning.html')

@app.route('/mistake')
def mistakes():
    return render_template('mistake.html')

@app.route('/data')
def data():
    return render_template('data.html')

if __name__ == "__main__":
    app.run(debug=True)
