""" 
Program: hw2.py
Programmed By: Caleb Sutton
Description: Implementation of KNN algorithm on MNIST
Trace Folder: Sutton728/hw2
"""

# imports
import sys     # used for csv input
import csv     # used for csv input
import math    # used for sqrt() in calculateDistance()
import time    # used to time algorithm

# program main
def main():
     # Check to ensure program is being used properly
     if (len(sys.argv) != 4):
          print("\nUsage: python hw2.py <trainingDataFile> <testDataFile> <k>\n")
          exit()

     # Start a timer for measuring program speed/performance
     startTime = time.time()

     # Variables used in main
     trainingData = []
     testData = []
     k = 0
     numCorrect = 0
     numIncorrect = 0
     accuracyRate = 0.0

     # Initialize training data, test data, and K
     trainingData = readCSV(sys.argv[1])
     testData = readCSV(sys.argv[2])
     k = int(sys.argv[3])

     # Print k
     print('\nK = ' + str(k) + '\n')

     # For each test sample in the testing data call the classify
     # sample function, keeping track of the number if correct
     # and incorrect samples
     for testSample in testData[1:]:
          if classifySample(trainingData[1:], testSample, k) == True:
               numCorrect += 1
          else:
               numIncorrect += 1

     # Compute the accuracy
     accuracyRate = numCorrect / (numCorrect + numIncorrect)
     round(accuracyRate, 4)

     # Print results
     print('\nAccuracy Rate: ' + str(accuracyRate * 100) + '%')
     print('Number of misclassified test samples: ' + str(numIncorrect))
     print('Total number of test samples: ' + str(numCorrect + numIncorrect) + '\n')

     # Print time elapsed
     endTime = time.time()
     print('Time Elapsed: ' + str(round((endTime-startTime), 3)) + 's\n')





# classifySample() function takes a set of training data, test
# sample, and k value as parameters and returns whether or 
# not the test samples computed class mathces its actual
# class
def classifySample(trainingData, testSample, k):
     # list for holding the nearest neighbors to the test sample
     nearestSamples = []

     # loop for each sample in the training data and calculate
     # its distance from the test sample and maintain a sorted
     # list of length k of the nearest neighbors
     for trainingSample in trainingData:
          # result is a dictionary used for keeping track of 
          # the nearest neighbors
          result = {}
          result['class'] = trainingSample[0]
          result['distance'] = calculateDistance(testSample[1:], trainingSample[1:])
          if result['distance'] == 0:
               result['vote'] = 1
          else:
               result['vote'] = 1/result['distance']

          # if there are less than k samples in the list append
          # results and then sort the array
          if len(nearestSamples) < k:
               nearestSamples.append(result)
               nearestSamples.sort(key = lambda sample: sample['distance'], reverse = False)
          # we only get to this else once the list contains k 
          # samples, which means we can iterate through and insert
          # the new sample in the correct postion, if its distance
          # is larger than all the samples it will be immediately
          # popped, otherwise the greatest distance will be popped
          else:
               i = 0
               for sample in nearestSamples:
                    if result['distance'] < sample['distance']:
                         nearestSamples.insert(i, result)
                         nearestSamples.pop()
                         break
                    i += 1

     # call organizational function calculateClass() to calculate
     # the class of the test sample using the nearestSamples list
     computedClass = calculateClass(nearestSamples)
     print('Desired Class: ' + str(testSample[0]) + ', Computed Class: ' + str(computedClass))

     # return true if the calculated class mathces the actual class
     # otherwise return false
     if int(computedClass) == int(testSample[0]):
          return True
     else:
          return False

# calculateDistance() takes two samples as parameters and returns
# the calculated euclidean distance bewteen the two
def calculateDistance(sample1, sample2):
     distance = 0

     for i in range(len(sample1)):
          square = int(sample1[i]) - int(sample2[i])
          distance += square * square

     return math.sqrt(distance)

# calculateClass() is a helper function to classifySample() it
# takes the nearestSamples list and then computes which class
# had the highest number of votes and returns it
def calculateClass(nearestSamples):
     votes = [0,0,0,0,0,0,0,0,0,0]
     highest = 0
     highestValue = 0

     for sample in nearestSamples:
          votes[int(sample['class'])] += sample['vote']

     for i in range(len(votes)):
          if votes[i] > highestValue:
               highestValue = votes[i]
               highest = i
     
     return highest

# readCSV() takes a file path to a .csv file as a parameter
# and returns a list of lists representing the csv file
def readCSV(filepath):
     data = []

     with open(filepath) as csvfile:
          readCSV = csv.reader(csvfile, delimiter = ',')
          
          for row in readCSV:
               data.append(row)

     return data

main()