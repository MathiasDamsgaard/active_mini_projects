from keras.models import Sequential
from keras.layers import Dense
import sklearn
import numpy as np

def task(args):
    identifier,Xtrain,ytrain,Xpool,committee_member = args
    print("Worker %s: starting task" % identifier)
    Xtr,ytr=sklearn.utils.resample(Xtrain,ytrain,stratify=ytrain)
            #Train committee member on Xtr,ytr
    committee_member.train(np.asarray(list(Xtr)).astype('float32'),[np.array(xi) for xi in ytr])
            #predict

    return committee_member.model.predict(Xpool.values)

class CommitteeMember:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=768*2, activation='relu'))
        self.model.add(Dense(1, activation='softmax'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=25, batch_size=64)


def oracle(idxs, df_pool):
    new_labels = []
    for i, idx in enumerate(idxs):
        ctx = df_pool.loc[idx]['context']
        q = df_pool.loc[idx]['question']

        # Check rate limit
        req_per_min += 1
        while req_per_min >= 19:
            time_stamp = time.time()
            if int(time.time() - last_time_stamp) > 60:
                last_time_stamp = time_stamp
                req_per_min = 0
            else:
                time.sleep(10)

        # Ask the oracle for label for the context and two questions
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system",
                     "content": "You are a system designed to label if a list of provided questions can be answered using ANY part of a provided context. You will always only reply in the following format for each question: `label: LABEL. LABEL should be 'y' if a question can be answered using the context and else 'n'"},
                    {"role": "user",
                     "content": "CONTEXT: ```The man in the house has a boy named Bob and a red car. He loves ice cream``` QUESTION: ```Is the boy named Jim?``` QUESTION: ```Does the man have a red book?```"},
                    {"role": "assistant", "content": "label: y\nlabel: n"},
                    {"role": "user", "content": f'CONTEXT: ```{ctx}``` QUESTION: ```{q["question"]}```'}
                ]
            )
        except Exception as e:
            print("OPENAI_ERROR:", str(e))
            continue

        # Parse the response to get labels
        res = completion.choices[0].message.content
        labels = re.findall(r'label: ([yn])', res)

        # If two labels are not found, assume something went wrong and skip this iteration
        if len(labels) != 1:
            print('[Error] labels not found in response:', res)
            continue

        new_labels.append(labels)

    return new_labels