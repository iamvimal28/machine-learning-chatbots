import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['!','?',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizing each word in patterns list
        word = nltk.word_tokenize(pattern)
        words.extend(word) #adding to existing list words
        #adding tuples in documents list
        documents.append((word,intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words)

#lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
#remove duplicates and sort words
words = sorted(list(set(words)))
#sort classes
classes = sorted(list(set(classes)))
#documents = combination of intent tag and pattern words
print (len(documents), "documents",documents)
#classes are intents tags
print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

# pickle.dump(words,open('words.pkl','wb'))
# pickle.dump(classes,open('classes.pkl','wb'))


#create training data
training = []

output_empty = [0] * len(classes)

#training set, bag of words for every sentence
for doc in documents:
    bag = []
    word_patterns = doc[0]
    #lemmatizing word patterns
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag,output_row])

#shuffle and make numpy array
# print("Training without shuffle",training)
random.shuffle(training)
# print("Training after shuffle",training)
training = np.array(training)
# print("Training numpy array",training)
#create training and testing lists X- patterns Y- intents
train_x = list(training[:,0])
# print("train_x",train_x)
train_y = list(training[:,1])
# print("train_y",train_y)
print("Training data is created")


#Creating Model

#The architecture of our model will be a neural network consisting of 3 dense layers. 
#The first layer has 128 neurons, the second one has 64 and the last layer will have the same 
#neurons as the number of classes. The dropout layers are introduced to reduce overfitting of the model. 
#We have used the SGD optimizer and fit the data to start the training of the model. 
#After the training of 200 epochs is completed, we then save the trained model using 
#the Keras model.save(“chatbot_model.h5”) function.

#Deep Neural Networks model
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# Compiling model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

#Training and saving the model
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5', hist)

print("model is created")










