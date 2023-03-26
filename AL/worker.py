from keras.models import Sequential
from keras.layers import Dense
import sklearn
import numpy as np

def task(args):
    identifier,Xtrain,ytrain,Xpool,committee_member,ypool_lab= args
    print("Worker %s: starting task" % identifier)
    Xtr,ytr=sklearn.utils.resample(Xtrain,ytrain,stratify=ytrain)
            #fit
    committee_member.train(Xtr.astype(np.float32), ytr)
            #predict

    return committee_member.model.predict(Xpool.values)

class CommitteeMember:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=768*2, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=25, batch_size=64)