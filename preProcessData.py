"""
This program pre-processes the water pump data from Driven Data/Taarifa by
converting csv files into lists and reducing the number of dimensions to encompass
only the important ones.

Authors: Emily Wu, Katrina Midgley
"""
import numpy as np
import csv
import os

def main():
    """
    This function re-writes the csv files and omits any unnecessary data. It then 
    converts all data files to a Data object.
    """
    readCSVFeatures("water_training_features.csv")
    readCSVLabels("water_training_labels.csv")
    data = Data("points", "labels", "att_names", 0, 59401)

def readCSVFeatures(data_file, dataSetType=None):
    """
    This function takes in a csv file of features and converts it to a text file. 
    """
    # saving all the attributes that we will use to classify the data
    with open(data_file) as f:
        omit_indexes = [0, 2, 8, 9, 12, 13, 21, 25, 26, 28, 29, 31, 33, 35, 37, 38]
        all = csv.reader(f)
        orig_atts = all.next()
        att_names = []
        for index in range(len(orig_atts)):
            if index not in omit_indexes:
                att_names.append(orig_atts[index])
        allData = []
        for l in all:
            line = []
            for i, element in enumerate(l):
                if i not in omit_indexes:
                    line.append(element)
            allData.append(line)

        # determining where to write the data points 
        if dataSetType == "unknown":
            of = open(os.path.join(".", "unknownPoints"), "w")
        else:
            of = open(os.path.join(".", "points"), "w")

        # writing the data
        for l in allData:
            for element in l:
                of.write("%s," %element)
            of.write("\n")
        of.close()

        of = open(os.path.join(".", "att_names"), "w")
            
        for element in att_names:
            of.write("%s\n" % element)
        of.close()

def readCSVLabels(data_file):
    """
    This function takes in a csv file of labels for each point and converts them
    into a text file.
    """
    with open(data_file) as f:
        all = csv.reader(f)
        all.next()
        labels = []
        for line in all:
            labels.append(line[1])
    of = open(os.path.join(".", "labels"), "w")
    for label in labels:
        of.write("%s\n" % label)
    of.close()

class Data(object):
    def __init__(self, pointFile, labelFile, attFile, start=None, end=None):
        """
        Data is an object containing all important variables for the dataset.
        They are as follows:

        - pointFile is a text file of all the data points to be used in CSV form
        - labelFile is a text file of all the labels to be used in CSV form
          corresponding to the points and each label seperated by a space
          attFile is a text file that contains the names of all the attributes
        - The member variables are:
          * x_data: a numpy array of all the point values
          * y_data: a numpy array of the point classifications
          * labels: a dictionary mapping attribute names to their possible values
        """
        with open(pointFile, 'r') as f:
            self.x_data = f.read()  # the points with their attribute-values converted to numbers
        self.x_data = self.x_data.split("\n")
        self.x_data.pop(-1)
        if start != None and end != None:
            for index in range(len(self.x_data)):
                self.x_data[index] = self.x_data[index].split(",")
                self.x_data[index].pop(-1)
            self.x_data = self.x_data[start:end]

        with open(attFile, 'r') as f:
            self.att_names = f.read() # the names of all the attribute names
        self.att_names = self.att_names.split("\n")
        self.att_names.pop(-1)

        if labelFile != None:
            with open(labelFile, 'r') as f:
                self.y_data = f.read() # the labels for each point as numbers

            self.y_data = self.y_data.split('\n')
            self.y_data.pop(-1)
                
            self.label_names = [] # the names of all the unique labels corresponding to their index numbers
            self.convertLabels()
            if start != None and end != None:
                self.y_data = self.y_data[start:end]
            
        self.atts = self.generateAtts() #list of all possible values for each attribute
        self.convertAttValues()
        self.convertToNumpy()

    def generateAtts(self):
        """
        This function looks through all the data points in self.points and gets all the
        unique values for each attribute.
        """
        atts = []
        for att in range(len(self.att_names)):
            att_list = []
            for point in range(len(self.x_data)):
                element = self.x_data[point][att]
                if element not in att_list:
                    att_list.append(element)
            atts.append(att_list)
        
        return atts

    def convertAttValues(self):
        """
        This function uses the self.atts list of lists to convert everything in self.x_data
        to their appropriate type.
        """
        continuous = [0,2,4,5,8,11,16]
        discrete = [1,3,6,7,9,10,12,13,14,15,17,18,19,20,21,22,23]

        for index in continuous:
            for point in self.x_data:
                point[index] = float(point[index])
            
        for att in discrete:
            for point in self.x_data:
                looking_for = point[att]
                new_label = self.atts[att].index(looking_for)
                point[att] = new_label

    def convertLabels(self):
        """
        This function converts all the labels to numerical values and stores
        them in self.y_data.
        """
        for label in self.y_data:
            if label not in self.label_names:
                self.label_names.append(label)
                
        for place in range(len(self.y_data)):
            self.y_data[place] = self.label_names.index(self.y_data[place])

    def convertToNumpy(self):
        for index in range(len(self.x_data)):
            convert = np.asarray(self.x_data[index])
            self.x_data[index] = convert
        sizeX = len(self.x_data)
        self.x_data = np.asarray(self.x_data).reshape(len(self.x_data), len(self.x_data[0]))


if __name__ == '__main__':
    main()
