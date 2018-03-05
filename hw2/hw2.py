import sys
import csv
import math

def main():
     if (len(sys.argv) != 4):
          print("\nUsage: python hw2.py <trainingDataFile> <testDataFile> <k>\n")
          exit()

     trainingData = []
     testData = []
     k = 0
     numCorrect = 0
     numIncorrect = 0
     accuracyRate = 0.0

     trainingData = readCSV(sys.argv[1])
     testData = readCSV(sys.argv[2])
     k = int(sys.argv[3])

     print('\nK = ' + str(k) + '\n')

     for testSample in testData[1:]:
          if classifySample(trainingData[1:], testSample, k) == True:
               numCorrect += 1
          else:
               numIncorrect += 1

     accuracyRate = numCorrect / (numCorrect + numIncorrect)

     print('\nAccuracy Rate: ' + str(accuracyRate * 100) + '%')
     print('Number of misclassified test samples: ' + str(numIncorrect))
     print('Total number of test samples: ' + str(numCorrect + numIncorrect) + '\n')


def readCSV(filepath):
     data = []

     with open(filepath) as csvfile:
          readCSV = csv.reader(csvfile, delimiter = ',')
          
          for row in readCSV:
               data.append(row)

     return data

def classifySample(trainingData, testSample, k):
     nearestSamples = []
     

     for trainingSample in trainingData:
          result = {}
          result['class'] = trainingSample[0]
          result['distance'] = calculateDistance(testSample[1:], trainingSample[1:])
          result['vote'] = 1/result['distance']

          if len(nearestSamples) < 1:
               nearestSamples.append(result)
          elif len(nearestSamples) <= k:
               nearestSamples.append(result)
               nearestSamples.sort(key = lambda sample: sample['distance'], reverse = False)
          else:
               i = 0
               for sample in nearestSamples:
                    if result['distance'] < sample['distance']:
                         nearestSamples.insert(i, result)
                         nearestSamples.pop()
                         break
                    i += 1

          #print('class = ' + str(trainingSample[0]) + ', distance = ' + str(distance))
          #print(nearestSamples)

     computedClass = calculateClass(nearestSamples)
     print('Desired Class: ' + str(testSample[0]) + ', Computed Class: ' + str(computedClass))

     if int(computedClass) == int(testSample[0]):
          return True
     else:
          return False


def calculateDistance(sample1, sample2):
     distance = 0

     for i in range(len(sample1)):
          square = int(sample1[i]) - int(sample2[i])
          distance += square * square

     return math.sqrt(distance)

def calculateClass(nearestSamples):
     votes = [0,0,0,0,0,0,0,0,0,0]
     highest = 0
     highestValue = 0

     for sample in nearestSamples:
          votes[int(sample['class'])] += sample['vote']

     for i in range(len(votes)):
          if votes[i] > highestValue:
               highest = i
     
     return highest

main()