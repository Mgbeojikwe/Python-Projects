"""
PREDICTIVE ANALYSIS:

A " Supervised-Learning " Linear classifier

This code model is suitable for scenrios of binary clasification with only one one feature.
for examples: The effect of customer's age on acceptance of company's product.
	       

Note the follwing :

(1) The following code is a linear clasifier for predictive analysis and thus not sutiable for non-linear condition

(2) It is used for 3 colunm data of which the first colunm houses the independent varirables(i.e the feature),
the second column houses the dependent variable , and the last colunm houses the class(group).
only two groups are considered . If your data has S/N, then your must only utilize data 
from the second column to the last 

(3) The classifier chosen in this case is a "linear classifier", a classifier 
that assumes a linear relationship between the "variable feature" and the "independent variable"

4) the classifier model is built in the while loop block by determining the suitable slope for such
linear relationship

5) this code-model finds the region that completely separates two groups of target classes


data requirements

1) two databases are needed for this code-model viz; the training data and test data

"""




import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys





def load_data(file_name):

    if file_name.endswith(".csv"):
        return pd.read_csv(file_name)
    
    elif file_name.endswith(".txt"):
        return pd.read_csv(file_name, sep=" ",header=0)

    elif file_name.endswith("xls"):
        return pd.read_excel(file_name)
    else:
        print("could not recognize file's extension")
        sys.exit()

class LinearClassifier:


    def __init__( self, training_filename):
        

        if training_filename.endswith(".csv") or training_filename.endswith(".txt") or training_filename.endswith(".xls"):
            self.__database=load_data(training_filename)
        
        elif training_filename.startswith("http"):
                
                filename = "filename" + training_filename[-4:] #taking extension into consideration

                import requests

                response = requests.get(training_filename, allow_redirects=True)
                with open(filename, mode="wb") as fh:
                    fh.write(response.content)

                self.__database = load_data(filename)
        else:
            print("file not recognizable")
            sys.exit()


        self.__tags=[]
        self.__groups = self.getting_group_arrays()
        self.__train = self.features()
        self.__m = self.plot_train()
        self.__field= list(self.__database.columns)

        
    def sorting_data(self):
        """ this function starts by reading the training data into variable "training_data". 
        then it now sorts the supplied training data. Sorting was row-wise and the sorting depends on the values of the feature. 
         It uses the .argsort() method of numpy, and returns the sorted training data
        """

        
        training_data= self.__database.values        
        
        sorted_train_data = training_data[training_data[ :, 0]. argsort()]
        return sorted_train_data

    def features(self):
        x_train = self.sorting_data()[:,0]
        y_train = self.sorting_data()[ :,1]

        return [ x_train, y_train]
        

    def getting_groups(self):
        """ since the training data has two groups which are in its last column, this method
        selects the two groups. since the data is now sorted, this method takes the first elements
        in the third column and takes the last element in the third column
        thus obtaining the two groups in the supplied data
        """
        L  = self.sorting_data()[0,2]  #getting the first group
        W = self.sorting_data()[-1,2]  #getting the second group

        self.__tags.append(L)
        self.__tags.append(W)

        return self.__tags

    def getting_group_arrays(self):
        """this method aims at getting the group arrays from the the sorted training data
        Firstly, it loops through the ordered data {self.sorting_data()} and on first 
        encountering the second group it breaks the loop.
        it then assign the elements in group self.t to variable "lesser" and assign the 
        array of group "self.w" to vzriable "greater" since such group have a greater value of 
        the variable feature. 
        """
        x=9 

        for i in range(len(self.sorting_data())): #iterating through the ordered trainid_data
            if self.sorting_data()[i,2] == self.getting_groups()[-1]:
                  x = i  #once the group self.w is encountered break away from the loop
                        #and assign the row value to vriable "x"
                  break 

        

        lesser = self.sorting_data()[ :x,: ]
        greater = self.sorting_data()[x: ,: ]

        return [lesser, greater]
        #return [ self.lesser, self.greater]
    
    @staticmethod  #I made the function a static method, because I want to utilize it only 
    def func(m, data_i): #within the code. I don't aim to utilize any instance arrtibute 
        return m* float(data_i) #or instance variable
    
    def plot_train(self):
        """
            firstly, this function obtains, from self.lesser and self.greater, the respective
            dependent and independent variable.
            Secondly it obtains the classifier( a linear classifier). To minimize the difficulty 
            of this, the method selects the the point of least length in array  "self.lesser"
            but selects the element of highest length in array "self.greater"
            
            the graphs were now plotted. in the plots we are to have 3 legends
        
        """ 
        x_lesser = self.__groups[0][ :,0]
        y_lesser = self.__groups[0][ :, 1]
        x_greater = self.__groups[-1][ :,0]
        y_greater = self.__groups [-1][ :,1]
        
        min_lesser =  min(y_lesser)#selecting the least length in "self.lesser"
        max_greater = max(y_greater)

        x,y=0,0  #assigning initial value to x and y

        for i in range( len(self.__groups[0])):
            lesser=self.__groups[0]
            if lesser[i][1] == min_lesser:
                x=i
                break 
        
        for i in range(len(self.__groups[-1])):
            greater = self.__groups[-1]
            if greater[i,1] == max_greater:
                y=i
                break 

        b= []

        b.append(lesser[ x,: ]) #appending the row with the least length in self.lesser
        b.append( greater[ y,:]) #appending the row with the highest length in self.greater

        m,average_error, how_close= 0.1, 0, 1E-2


        while( average_error < how_close):

            #this loop finds the region that completely separates the two classes
            #The region is found in stepwise movement

            t_predicted = self.func(m,b[0][0])
            error1 = b[0][1] - t_predicted

            w_predicted = self.func(m, b[1][0])
            error2 = b[1][1]- w_predicted

            average_error=  (error1 + error2)/2

            change_in_m = average_error/b[0][1]

            m = m + change_in_m
        
        #on leaving the while loop we now have the suitable slope for the classifier 
        
        

        

        return m
    
    def test(self, test_data):
            """ this method tests the classifier on a supplied data.
            the supplied data is first made an array, its variable feature and independant variable
            obtained. A plot is now made along with the the classifier . The plots shows 
            the points for the training data, the test data, and the classifier ( linear classifier)

            THe resulting plot is saved in file "test.png"
            """ 
            test_data = pd.read_csv(test_data).values  
 
            x_test_data = test_data[ :,0]
            y_test_data = test_data[ :,1]

            classifier = [self.func(self.__m, i) for i in x_test_data]

            lesser  =  self.__groups[0]
            greater = self.__groups[-1]

            x_lesser = lesser[ :,0]
            y_lesser = lesser[ :,1]
            x_greater = greater[:,0]
            y_greater = greater[:,1]


            plt.scatter(x_test_data, y_test_data, label="test set")
            plt.scatter(x_lesser, y_lesser, label="catepillar" if f"{self.__tags[0]}"== "C" else f"{self.__tags[0]}")
            plt.scatter(x_greater,y_greater,label= "lady bird" if f"{self.__tags[-1]}"== "L" else f"{self.__tags[-1]}")
            plt.plot(x_test_data, classifier, label="classifier")
            
            plt.legend()
            #plt.xlabel("independent variable")
            plt.xlabel(f"{self.__field[0]}")
            plt.ylabel("dependent variable")
            plt.ylabel(f"{self.__field[1]}")
            plt.savefig("test.png", format="png")
            plt.show()


new_module=".................................................................."



"""
The below two datasets (bugs-train.csv and bugs-test.csv) are remote data that can be used as examples in training the model
"""
training_data="http://54.243.252.9/ce-5319-webroot/2-Exercises/ES-1/bugs-train.csv"#"bugs-train.csv" #-that is, the trianing data  
data2 ="http://54.243.252.9/ce-5319-webroot/2-Exercises/ES-1/bugs-test.csv" #"bugs-challenge.csv"  #-that is, the test data


#data = np.array(data)

obj = LinearClassifier(training_data)



obj.test(data2)
