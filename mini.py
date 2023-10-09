
from sklearn.linear_model import LinearRegression
#training the model
lr=LinearRegression()
lr.fit(X_train,y_train)
#applying the model to make a prediction
y_lr_train_pred=lr.predict(X_train)
y_lr_test_pred=lr.predict(X_test)
print(y_lr_train_pred,y_lr_test_pred)
#evaluating model
from sklearn.metrics import mean_squared_error,r2_score
lr_train_mse=mean_squared_error(y_train,y_lr_train_pred)#training mse
lr_train_r2=r2_score(y_train,y_lr_train_pred)#training r2
lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)#testing mse
lr_test_r2=r2_score(y_test,y_lr_test_pred)#testing r2
print("LR MSE(train)" , lr_train_mse)
print("LR r2(train)", lr_train_r2)
print("LR MSE(test)" , lr_test_mse)
print("LR r2(test)", lr_test_r2)
lr_results=pd.DataFrame(["Linear regression" , lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()#forming dataframe for mean square and r2
lr_results.columns=["methods","training mse","training r2","testing mse","testing r2"]#changing the columns name
