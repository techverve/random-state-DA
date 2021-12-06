'''------- EXTERNAL IMPORTS -------'''
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb 
from sklearn.linear_model import  LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import  metrics
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

def train_models(df, rs_limit):
    model_linear = LinearRegression()
    model_xgb = xgb.XGBRegressor(n_estimators=5000)
    model_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    random_states = [i for i in range(10, rs_limit,10)]
    errors_linear, errors_xgb, errors_svr  = [], [], []
    scores_linear, scores_xgb, scores_svr = [], [], []
    for rs in random_states:

          X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,0:-1], df.iloc[:,-1], test_size=0.2, random_state=rs, shuffle = True)
          model_linear.fit(X_train, Y_train)
          Y_linear = model_linear.predict(X_test)
          scores_linear.append(cross_val_score(
          model_linear, df.iloc[:,0:-1], df.iloc[:,-1], cv=5))
          
          model_xgb.fit(X_train, Y_train)
          Y_xgb = model_xgb.predict(X_test)
          scores_xgb.append(cross_val_score(
          model_xgb, df.iloc[:,0:-1], df.iloc[:,-1], cv=5))


          model_svr.fit(X_train, Y_train)
          Y_svr = model_svr.predict(X_test)
          scores_svr.append(cross_val_score(
          model_svr, df.iloc[:,0:-1], df.iloc[:,-1], cv=5))
          
          errors_linear.append(mean_squared_error(Y_test, Y_linear))
          errors_xgb.append(mean_squared_error(Y_test, Y_xgb))
          errors_svr.append(mean_squared_error(Y_test, Y_svr))
    return errors_linear, errors_xgb, errors_svr, scores_linear, scores_xgb, scores_svr


def infer(df):
	print("Distribution")
	sns.distplot(df['col_4']);
	errors_linear, errors_xgb, errors_svr, scores_linear, scores_xgb, scores_svr = train_models(df)
	
	f, axs = plt.subplots(2,2,figsize=(15,15))
	axs[0,0].plot(random_states, errors_svr)
	axs[0,0].set_xlabel('random_state values')
	axs[0,0].set_ylabel('SVR errors')
	 
	# displaying the title
	axs[0,0].set_title('SVR - random_state vs mse error')

	axs[0,1].plot(random_states, errors_linear)
	axs[0,1].set_xlabel('random_state values')
	axs[0,1].set_ylabel('Linear regression errors')
	 
	# displaying the title
	axs[0,1].set_title('Linear - random_state vs mse error')

	axs[1,0].plot(random_states, errors_xgb)
	axs[1,0].set_xlabel('random_state values')
	axs[1,0].set_ylabel('XGboost errors')
	 
	# displaying the title
	axs[1,0].set_title('XGBoost - random_state vs mse error')
	plt.show()
       
    print('Pearson coefficient')
 	print(pearsonr(errors_linear,errors_svr))
 	print("Cross validation score for different values of random_state for Linear regression - ", scores_linear)
	print("Cross validation score for different values of random_state for SVR - ", scores_svr)
	print("Cross validation score for different values of random_state for XGBoost - ", scores_xgb)
 	
 if __name__ == '__main__':
 	 df1 = pd.read_csv('100rows_dataset')
 	 df2 = pd.read_csv('500rows_dataset')
 	 df3 = pd.read_csv('1000rows_dataset')
 	 df4 = pd.read_csv('10000rows_dataset')
 	 df5 = pd.read_csv('100000rows_dataset')
	 infer(df1)
	 infer(df2)
	 infer(df3)
	 infer(df4)
	 infer(df5)
 
