"""This source code assumes that: 
1) The data is clean and elements in each column of the pandas dataframe are all strings
2) The label is made the last column in the dataset
"""

import pandas as pd
import xlrd
import numpy as np
import sys
import tensorflow as tf


from tensorflow.python.framework import indexed_slices
def load_data(file_name):
    if file_name.endswith(".csv"):
        return pd.read_csv(file_name)

    elif file_name.endswith("xls"):
      book = xlrd.open_workbook(file_name) 
      return pd.read_excel(book)
    
    elif file_name.endswith("txt"):
        return  pd.read_csv(file_name,sep=" ",header=0)
    else:
      pass
    

class Encoding:

  __slots__ = ("file_name")


  def __init__(self,file_name): 

    #let us instead consider having the file name as instance attribute 
    file_name = str(file_name)

    self.file_name=file_name 
    self.__dataframe=0
    
    
    
    if self.file_name.endswith("csv") or self.file_name.endswith("xls") or self.file_name.endswith("txt"):
          self.__dataframe = load_data(self.file_name)
    
    elif self.file_name.startswith("http"):  #if remote file

          url=self.file_name
          filename="filename"+url[-4:]  #taking the file's extention into consideration
          
          import requests 
          
          response = requests.get(url, allow_redirects=True)
          with open(filename, mode="wb") as fh:
            fh.write(response.content)
          
          self.__dataframe = load_data(filename)
          
    
    else:
      print("could not recognize the extension of the file")
      sys.exit()

    
  
    self.__columns=list(self.__dataframe.columns.values) #creating a list of the columns names with the last column being the training label
                                                      
   
   
    
    self.__words_lists = [ [] for column in self.__columns]  #a 2-D list where by each list will later contain the unique 
                                                          #words in their respective coulmns. the number of empty lists
                                                          #is equal to the number of columns in the dataframe. The last column 
                                                          #is taken to be the vector of labels
    
    self.__indexes_lists= [ [] for index in self.__columns]  #this is also a 2D list of whereby each list will be used to store 
                                                  #the indexes of each unique words obtained from the respective columns       
    


    self.__de_encoded_words=[]
    self.__encoded_words=[]
    self.__sentence=" "
    self.__sc_x=0
    self.__sc_y=0
    self.__ann = tf.keras.models.Sequential()


  def encoding_training_data(self):
    """
    This method is used in encoding the entire pandas dataframe i.e (features and label).
    

    this method does the following:
    (a) It extracts all the element/words in a column without repetition via the set() function, and forms a list. The resulting
    list is assigned to an object "word_list" that represent each member of the list "self.words_lists"

    (b) For each column, with the aid of the Zip() function, the function replaces each word in the column by an integer
    hence encoding the column
    (c) Then for each column, it creates a container of indexes such that the indexes have sames pythonic positions as there original words in
    "self.list_of_word" respectively for each column
    (d) the methods fill empty spaces i.e np.nan, in the each column with zeros
    (e) Encoding starts with integer 1, since the plan is to fill empty spaces i.e np.nan,  with zeros
    """
    for index,column_name in enumerate(self.__columns):
        
        self.__dataframe[column_name] = self.__dataframe[column_name].fillna(0)  #filling empty spaces i.e np.nan with zero

        word_list= list(set(self.__dataframe[column_name]))                           #this line gets all words(i.e elements) in a given column without repetition
                                                                                      #+ making each word indexable then assign them to an object "entire_datacells"(i.e this object 
                                                                                      #+ represents the list of words its respective column

        self.words_lists[index] = [word.strip() for word in word_list]

    
    #Now that we have filled the each word_list with its respective words, we will replace
    #each word with a number, thus encoding each word
    for column_name , word_list, index_l in  zip(self.__columns, self.__words_lists,range(len(self.words_lists))):
     
          for index, word in enumerate(word_list,1):
              if  word != str(0):                 #ignoring zero 
                  self.__dataframe[column_name] = self.__dataframe[column_name].replace(word,index)  #encoding the words in each colunm
          
                  self.indexes_Lists[index_l].append(index)                                #appending the indexes to a list of indexes reserved for each column. this list is needed in de-encoding of words

              else:
                self.indexes_lists[index_l].insert(index,0)                                 #inserting zero(0) at same index position it is located in word_list

                                                                 
    

    self.__words_lists =[[word.lower()  for word in word_list]  for word_list in self.__words_lists]  #converting the words to lower case
    return True

  
  def train_model(self):

      """This method uses  Neural Network in training the model, compared result obtained when feature scaling is applied to only features and (features and label) 
      also compare the result obtained when you didn't standardize
      
      The first column to (n-1) column are used as features while the last column used as label.
      Both features and label are scaled using the StandardScalar class
      The data was used to train a neural network that has two hidden layers. The ReLU fucntion was used as the activation function in hidden layers,
      +while the linear function was used as the activation for the outer layer. Only one Neuron/unit was used for the outer layer since we are carrying-out a regression analysis

      """

      result=self.encoding_training_data()                                                                       #running the encoder() method

      x=self.__dataframe.iloc[:,:-1].values
      y=self.__dataframe.iloc[:,-1:].values                                     
      
      # from sklearn.model_selection import train_test_split

      # x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=1) 

      from sklearn. preprocessing import StandardScaler 

      self.__sc_x =  StandardScaler()
      # x_train = self.__sc_x.fit_transform(x_train)
      # x_test = self.__sc_x.transform(x_test)                                                        #applying the standard deviation used for "x_train" to that for "x_test"

      self.__sc_y = StandardScaler()                                                                   
      #y_train = self.__sc_y.fit_transform(y_train) 

      #there is a possibility that we will train the model using the entire data
      #if yes, then:
      x_train = self.__sc_x.fit_transform(x)
      y_train = self.__sc_y.fit_transform(y) 

      num_of_features = len(self.__columns)-1
      num_of_neurons = int((num_of_features + 1)/2)
      
      
      self.__ann.add(tf.keras.layers.Dense(units=num_of_neurons,activation="relu"))                         #adding the first hidden layer
      self.__ann.add(tf.keras.layers.Dense(units=num_of_neurons,activation="relu"))                         #adding the second hidden layer
      self.__ann.add(tf.keras.layers.Dense(units=1,activation="linear"))                                    #Output layer uses the "linear" activation fucntion since we are doing regression analysis
      
       
      self.__ann.compile(optimizer="adam",loss="mean_absolute_error", metrics=["mean_absolute_error"])
      
      self.__ann.fit(x_train,y_train,batch_size=32, epochs=300)

      return True

  

  
  def  encode_new_sentence(self):
      
        """
           This methods starts by running the train_model() method
        => The method then encodes features that are used to test the extensibility of the Machine Learning model.
        => Recall that a row corresponds to one training data.
        => The feature is supplied as a 2-dimensional list. where each list represents a single 
           training data with strings  as its elements.
        => The strings are encoded similar to their name-sake in the training dataframe
        =>  strings that are not present in training feature columns are encoded as zero.

        => recall that: 
        
          a training row = [ [x1, x2, x3, x4, ....,Xn]   
                             ]                      where Xi denotes a string corresponding to feature-column i

        => Thus each word in the supplied sentence is compared to words in its respective column
        """
        
        result=self.train_model()
        
        
        
        self.__encoded_words.clear()
        

        self.__sentence =input("enter incomplete sentence: ").lower()
        
        words = self.__sentence.split()                                                       #splitting the sentence based on white space 
        
        num_of_features=len(self.__columns)-1                                                               #number of feature columns

        if len(words) > num_of_features :                                                                   #if number of words in sentence is greater than the number of feature columns
            sum="a"
            for word in words[num_of_features-1:]:
                  sum=sum+word+" "                                                            #concatenating the excess words whose index positions are greater than the number of training features 
            sum    =  sum[1:]                                                                 #removing "a"
            words  = words[:num_of_features - 2]                                                              #truncating the list of words
            words.append(sum)                                                                 #appending the concatenated words to "words" list

        elif  len(words) < n: 
              for i in range(n- len(words)):
                    words.append(str(0))                                                      #lagging the words with zeros to ensure it matches the number of features columns
        else :
            words=words
        
        
        for training_words_list, training_indexes_list, word in zip(self.__words_lists,self.__indexes_lists,words):

                if word in training_words_list: 

                      for index, element in enumerate(training_words_list):
                              if word == element:
                                      self.__encoded_words.append(training_indexes_list[index])

                else:
                  self.__encoded_words.append(0)                                                      #words not present in training feature is encoded as zero
        


        self.__encoded_words=self.__sc_x.transform([self.__encoded_words])                            #scaling the encoded data using the standard deviation of the that used in scaling the training set


        return True
     
         
   
  
  def de_encoding_predicted_label(self, predicted_label):
    
    """ 
    Firstly, the predicted label (which is a scaled number) is de-scaled using the "inverse_transform()" method.
    The resulting number is now searched in  "self.__indexes_lists" container.
    If found, it is returned. 

    """

    predicted_label=self.__sc.y.inverse_transform(predicted_label)
    
    self.__words_lists,self.__indexes_lists,self.__columns = self.__encoder()

    for indexes_list, words_list in zip(self.__indexes_lists,self.__words_lists):
        if predicted_label in indexes_list:
              for index, number in enumerate(indexes_list):
                    if predicted_label == number:
                            predicted_word = words_list[index]    
                            return predicted_word

  def predict_label(self):
      """
      This method starts by predicting the label for the incomplete sentence. 
      The predicted label, which is in scaled format, is de_scaled and the word predicted using the "de_encoding_predicted_label()" method

      """
      
      result = self.encode_new_sentence()
      
      predicted_code = self.__ann.predict(self.__encoded_words)

      predicted_word = self.de_encoding_predicted_label(predicted_code)
      return predicted_word
  
  
