#Supervised Machine Learning from free code camp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

#Step 1: acquire data from "archive.ics.uci.edu"
cols=["flength", "fwidth", "fsize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df=pd.read_csv("magic04.data", names=cols)
df["class"]=(df["class"]=="g").astype(int) # Convert class type into integer 1 or 0, "g"->1, "h"->0

#Step 2: Draw a visualized diagram of the data
for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color="blue", label="gamma", alpha=0.7, density=True) #alpha sets the transparency
    plt.hist(df[df["class"]==0][label], color="red", label="hadron", alpha=0.7, density=True) #density makes different sizes of data comparable
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
#Step 3: Separate the data into three sections: training data, validation data, and testing data
train, valid, test=np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

#Step 4: Scale dataset so that the mean or standard deviation does not effect the model
def scale_dataset(dataframe, oversample=False):
    x=dataframe[dataframe.columns[:-1]].values # Extract data except the "class" column
    y=dataframe[dataframe.columns[-1]].values # Extract data from the "class" column

    scaler=StandardScaler()
    x=scaler.fit_transform(x) # Scale the value in x

    if oversample:
        ros=RandomOverSampler()
        x,y=ros.fit_resample(x,y) # Take the fewer "class" and resample them so that the two datasets(gamma & hadron) became the same in number

    data=np.hstack((x,np.reshape(y,(-1,1)))) # put the data together again, Note: reshape y into an array of the shape len(y) * 1 

    return data, x, y

train, x_train, y_train=scale_dataset(train, overSample=True)
valid, x_valid, y_valid=scale_dataset(valid, overSample=False)
test, x_test, y_test=scale_dataset(test, overSample=False)

#Step 5(Real Machine Learning)Neural Network [Tensorflow]
import tensorflow as tf
def train_model(x_train,y_train,num_nodes,dropout_prob,lr,batch_size,epochs):
    nn_model=tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes,activation='relu',input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes,activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='binary_crossentropy',
                    metrics=['accuracy'])
    # L1 Loss Function: sum(|yreal - ypredicted|)
    # L2 Loss Function: sum((yreal - ypredicted)^2)
    # Binary Cross-Entropy Loss: -1/N * sum(yreal * log(ypredicted) + (1 - yreal) * log(1 - ypredicted))
    
    #Step 6: put our data in neural network
    history=nn_model.fit(
        x_train,y_train,epochs=epochs,batch_size=batch_size,validation_split=0.2,verbose=0
    )

    return nn_model,history


#Step 7: Plot Loss and Accuracy
def plot_history(history):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    ax1.plot(history.history['loss'],label='loss')
    ax1.plot(history.history['val_loss'],label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary cross entropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'],label='accuracy')
    ax2.plot(history.history['val_accuracy'],label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

#Step 8: Compare different models for optimization
least_val_loss=float('inf')
least_loss_model=None
epochs=100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, learning rate {lr}, batch size {batch_size}")
                model,history=train_model(x_train,y_train,num_nodes,dropout_prob,lr,batch_size,epochs)
                plot_history(history)
                val_loss=model.evaluate(x_valid,y_valid)
                if val_loss < least_val_loss:
                    least_val_loss=val_loss
                    least_loss_model=model

#Step 9: Predict using test data
y_pred=least_loss_model.predict(x_test)
y_pred=(y_pred>0.5).astype(int).reshape(-1,)
print(classification_report(y_test,y_pred))



