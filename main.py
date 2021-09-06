import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

from nsepy import get_history
from datetime import datetime
from matplotlib import pyplot as plt

def menu():
    
    print("")
    print("***************************************************")
    print("***************************************************")
    print("************** BTC-USD PREDICTION *****************")
    print("******************* by Felix **********************")
    print("***************************************************")
    print("***************************************************")
    print("")

    FileName = str(input("File Name CSV : "))
    TimeSteps = int(input("Time Steps : "))
    FutureTimeSteps = int(input("Future Time Steps : "))
    BatchSize = int(input("Batch Size : "))
    Epochs = int(input("Epochs : "))

    return FileName,TimeSteps, FutureTimeSteps, BatchSize, Epochs

def main():
    inputAn = menu()
    FilenameCSV = inputAn[0]
    TimeSteps = inputAn[1]
    FutureTimeSteps = inputAn[2]
    TestingRecords = FutureTimeSteps
    BatchSize = inputAn[3]
    Epochs = inputAn[4]

    filename = FilenameCSV+".csv"
    StockData = pd.read_csv(filename)
    #print(df)
    #df = StockData
    #print(df)
    #df[['Open','Close']].plot()
    #plt.show()
    
    StockData['TradeDate']=StockData['Date']
    #StockData.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)
    #plt.show()

    FullData=StockData[['Close']].values
    # Feature Scaling for fast training of neural networks
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Choosing between Standardization or normalization
    #sc = StandardScaler()
    sc=MinMaxScaler()

    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)
    #X=FullData

    #print('### After Normalization ###')
    #X[0:5]
    # Printing the last 10 values
    print('Original Prices')
    print(FullData[-TimeSteps:])

    print('###################')

    # Printing last 10 values of the scaled data which we have created above for the last model
    # Here I am changing the shape of the data to one dimensional array because
    # for Multi step data preparation we need to X input in this fashion
    X=X.reshape(X.shape[0],)
    print('Scaled Prices')
    print(X[-TimeSteps:])

    # Multi step data preparation

    # split into samples
    X_samples = list()
    y_samples = list()

    NumerOfRows = len(X)
    #TimeSteps=10  # next few day's Price Prediction is based on last how many past day's prices
    #FutureTimeSteps=5 # How many days in future you want to predict the prices

    # Iterate thru the values to create combinations
    for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i:i+FutureTimeSteps]
        X_samples.append(x_sample)
        y_samples.append(y_sample)

    ################################################

    # Reshape the Input as a 3D (samples, Time Steps, Features)
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    print('### Input Data Shape ###') 
    print(X_data.shape)

    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data=np.array(y_samples)
    print('### Output Data Shape ###') 
    print(y_data.shape)

    # Choosing the number of testing data records
    #TestingRecords=5

    # Splitting the data into train and test
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    y_train=y_data[:-TestingRecords]
    y_test=y_data[-TestingRecords:]

    #############################################
    # Printing the shape of training and testing
    print('\n#### Training Data shape ####')
    print(X_train.shape)
    print(y_train.shape)

    print('\n#### Testing Data shape ####')
    print(X_test.shape)
    print(y_test.shape)

    # Visualizing the input and output being sent to the LSTM model
    # Based on last 10 days prices we are learning the next 5 days of prices
    for inp, out in zip(X_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)

    # Defining Input shapes for LSTM
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    print("Number of TimeSteps:", TimeSteps)
    print("Number of Features:", TotalFeatures)

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = TimeSteps, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

    # Adding the Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = FutureTimeSteps, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

    # Adding the Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = FutureTimeSteps, activation = 'relu', return_sequences=False ))


    # Adding the output layer
    # Notice the number of neurons in the dense layer is now the number of future time steps 
    # Based on the number of future days we want to predict
    regressor.add(Dense(units = FutureTimeSteps))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ###################################################################

    import time
    # Measuring the time taken by the model to train
    StartTime=time.time()

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, batch_size = BatchSize, epochs = Epochs)

    EndTime=time.time()
    print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

    # Making predictions on test data
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    print('#### Predicted Prices ####')
    print(predicted_Price)

    # Getting the original price values for testing data
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)
    #print('\n#### Original Prices ####')
    #rint(orig)

    #print('\n####Join Prices ####')
    #print('####Last 60 Predicted Prices ####')
    #joinOrig = np.concatenate((orig[0], orig[1]), axis=None)
    #print(joinOrig)
    for i in range(len(orig)):
        Prediction=predicted_Price[FutureTimeSteps-1]
        Original=orig[FutureTimeSteps-1]
        #print('Prediction')   
        #print(Prediction)

    plot1 = plt.figure("Test Accuracy",figsize=(15,4))
    #plt.figure(figsize=(10,3))
    ax = plt.axes()
    ax.set_facecolor("grey")
    # Visualising the results
    plt.plot(Prediction, color = 'blue', label = 'Predicted Price')
    plt.plot(Original, color = 'yellow', label = 'Original Price')

    plt.title('### Accuracy of the predictions:'+ str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2))+'% ###')
    plt.xlabel('Date')
                    
    startDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1)
    endDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1) + FutureTimeSteps
    TotalRows=StockData.shape[0]
            
    #print('10 harga terakhir')
    #print(predicted_Price[-2:])
    #print(range(FutureTimeSteps), StockData.iloc[TotalRows-endDateIndex : TotalRows-(startDateIndex) , :]['TradeDate'])
    #print(FutureTimeSteps)
            
    plt.xticks(range(FutureTimeSteps), StockData.iloc[TotalRows-endDateIndex : TotalRows-(startDateIndex) , :]['TradeDate'])
    plt.ylabel('Price (BTC-USD)')

    plt.gca().xaxis.set_tick_params(rotation = 40)  
    plt.legend()
    plt.gcf().autofmt_xdate()

    #plt.show()

    # Making predictions on test data
    Last10DaysPrices=orig[-2:]
    print("###### Last10DaysPrices ########")
    print(Last10DaysPrices)
    # Reshaping the data to (-1,1 )because its a single entry
    Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)
    # Scaling the data on the same level on which model was trained
    X_test=DataScaler.transform(Last10DaysPrices)

    NumberofSamples=1
    TimeSteps=X_test.shape[0]
    NumberofFeatures=X_test.shape[1]
    # Reshaping the data as 3D input
    X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)

    # Generating the predictions for next 5 days
    Next5DaysPrice = regressor.predict(X_test)

    # Generating the prices in original scale
    Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)
    #Next5DaysPrice

    #print('NEXT 5 Days')

    #print(range(FutureTimeSteps+5), StockData.iloc[TotalRows-endDateIndex : TotalRows+5-(startDateIndex) , :]['TradeDate'])
    print(Next5DaysPrice)

    import matplotlib.dates as mdates
    import datetime as dt
    Date=StockData['Date']
    now = max(Date)

    now = datetime.fromisoformat(now) + dt.timedelta(days=1)
    then = now + dt.timedelta(days=FutureTimeSteps)
    days = mdates.drange(now,then,dt.timedelta(days=1))

    plot2 = plt.figure('Prediction',figsize=(15,4))
    ax = plt.axes()
    ax.set_facecolor("grey")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_tick_params(rotation = 40)  
    plt.title('### Prediction For Next '+ str(FutureTimeSteps) +' Days ###')
    plt.xlabel('Date')
    plt.ylabel('Price (BTC-USD)')

    plt.plot(days,Next5DaysPrice[0], color="blue")
    plt.gcf().autofmt_xdate()
    plt.show()

main()