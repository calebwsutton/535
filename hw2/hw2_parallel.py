""" 
Program: hw2_parallel.py
Programmed By: Caleb Sutton
Description: Parallelized implementation of KNN algorithm on MNIST
Trace Folder: Sutton728/hw2
"""

# imports
import sys     # used for csv input
import csv     # used for csv input
import math    # used for sqrt() in calculateDistance()
import time    # used to time algorithm
import multiprocessing as mp  # used to implement multiprocessing

# program main
def main():
     # Check to ensure program is being used properly
     if (len(sys.argv) != 4):
          print("\nUsage: python hw2_parallel.py <trainingDataFile> <testDataFile> <k>\n")
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

     # print K
     print('\nK = ' + str(k) + '\n')

     # variable specific to multiprocessing
     que = mp.Queue()
     cpuCount = mp.cpu_count()
     samplesPerCore = [0 for i in range(cpuCount)]
     processes = []

     # calculate the number of samples to be classified by
     # each core
     for i in range(cpuCount):
          samplesPerCore[i] = len(testData[1:]) / cpuCount
          if i < len(testData[1:]) % cpuCount:
               samplesPerCore[i] += 1

     # start each process using the classifySamples() function
     # giving it the training data, and a chunk of the test samples
     startIndex = 1
     for i in range(cpuCount):
          samples = testData[int(startIndex) : int(startIndex + samplesPerCore[i])]
          p = mp.Process(target = classifySamples, args = [trainingData[1:], samples, k, que], name = 'classify_process_' + str(i))
          processes.append(p)
          p.start()
          startIndex += samplesPerCore[i]

     # wait for processes to finish
     for p in processes:
          p.join()

     # get the results from each process using the que
     for i in range(que.qsize()):
          result = que.get()
          if result == True:
               numCorrect += 1
          else:
               numIncorrect += 1

     # compute the accuracy
     accuracyRate = numCorrect / (numCorrect + numIncorrect)
     round(accuracyRate, 4)

     # print the results
     print('\nAccuracy Rate: ' + str(accuracyRate * 100) + '%')
     print('Number of misclassified test samples: ' + str(numIncorrect))
     print('Total number of test samples: ' + str(numCorrect + numIncorrect) + '\n')

     # print the time elapsed
     endTime = time.time()
     print('Time Elapsed: ' + str(round((endTime-startTime), 3)) + 's\n')





# classifySamples() function takes a set of training data, test
# samples, k value, and a que as parameters. it adds a true to
# the que if it correctly classifies a sample, and a false
# if it incorrectly classifies a sample
def classifySamples(trainingData, testSamples, k, que):
     # loop through each sample in the list of test samples
     for testSample in testSamples:
          # list for holding nearest neighbors
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
                    result['distance'] = 999999
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

          # add true if the calculated class mathces the actual class
          # otherwise add false
          if int(computedClass) == int(testSample[0]):
               que.put(True)
          else:
               que.put(False)

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

if __name__ == "__main__":
     main()