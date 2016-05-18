#Using machine learning to determine the status of Tanzanian water pumps based on data from Taarifa

AUTHORS: Emily Wu and Katrina Midgley 

DATE: May 2016

SUMMARY:
We were interested in using Machine Learning methods to correctly classify
water pumps in Tanzania as working, needing repair, or broken based on data
collected for each water pump in the country. The data were obtained from 
DrivenData.org, which obtained the data from Taarifa. Our program
processes the data and then runs multiple supervised, unsupervised and ensemble
learning techniques. Our objective was to find a method and find parameters that 
minimized our error rate.

FILES:

classifier.py
- Contains superverised and unsupervised learning methods that classify the 
  data and outputs error rates for each method

preProcessData.py
- Contains methods which read in the csv files from DrivenData.org for their
  competition on water pump classfication
- Manipulates the data to remove unnessary dimensions of the water pump data
- Contains a Data object that contains all the attributes needed of this data

INSTRUCTIONS TO RUN:

1) Update source files & directories in preProcessData.py and classifier.py

2) Run preProcessData.py to clean the data and produce the necessary files

3) Run classifier.py to run the classifications

NECESSARY LIBRARIES:

This code requires multiple methods from python's sklearn, scipy and numpy libraries

SOURCE:

https://www.drivendata.org/competitions/7/data/


