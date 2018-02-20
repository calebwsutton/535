# CSC 535 Homework 1 Part 1
# Authors: Caleb Sutton, Lyubov Sidlinskaya, Josiah McGurty

import pandas as pd
import numpy as np
import copy
import sys
import csv

training_data_input = [
	({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
	({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
	({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
	({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
	({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
	({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
	({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
	({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
	({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
	({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
	({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
	({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
	({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
	({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
] 

test_sample1 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"} #True
test_sample2 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"} #False
test_sample3 = {"level" : "Intern"} # True
test_sample4 = {"level" : "Senior"} # False 
test_sample5 = {"level" : "Mid"} # True

def read_data(training_data):
	training_data_new = []
	attributes = {}

	# build a dictionary of attributes to arrays. the arrays will hold the possible values of each attribute
	for key, _ in training_data[0][0].items():
		attributes[key] = []

	# create a new set of training data where class is now part of each dictionary
	for record in training_data:
		new_record = record[0]
		new_record['class'] = record[1]
		training_data_new.append(new_record)

	# create a data frame from the new set of training data
	training_data_frame = pd.DataFrame(training_data_new)

	# find all the possible values for each attribute and update the dictionary arrays
	for key, _ in attributes.items():
		values = []
		for value in training_data_frame[key].unique():
			values.append(value)
		attributes[key] = values

	# testing
	final_dt = (build_tree(training_data_frame, attributes))
	print (final_dt)
	print ("\n")
	print(test_sample1)
	print(classify(final_dt, test_sample1))
	print ("\n")
	print(test_sample2)
	print (classify(final_dt, test_sample2))
	print ("\n")
	print(test_sample3)
	print (classify(final_dt, test_sample3))
	print ("\n")
	print(test_sample4)
	print (classify(final_dt, test_sample4))
	print ("\n")
	print(test_sample5)
	print (classify(final_dt, test_sample5))

# build_tree() function accepts a pandas dataframe and dictionary of attribute names to arrays of possible values as parameters
# 	for example attributes could look like {'level' : ['junior', 'senior', 'mid'], phd : ['yes', 'no']} 
# 	training_data_frame should be a pandas dataframe where the class attribute has been added
# build_tree() returns a decision tree
def build_tree(training_data_frame, attributes):
	num_true = 0 	# used as a count for the number of (class == true) training data samples
	num_false = 0	# used as a count for the number of (class == false) training data samples
	entropies = {} # dictionary used to store the possible attributes to split on and their respective entropies
	tree = () 	# return value (possibly unnecessary)

	# count number of values in each class
	for value in training_data_frame['class']:
		if value == True:
			num_true += 1
		else :
			num_false +=1

	# if the vale num_true or num_false is zero then that means all the values must be the opposite
	if num_true == 0:
		return False
	elif num_false == 0:
		return True

	# if there are no attributes left return the most common class
	if len(attributes) == 0:
		if num_false > num_true:
			return False
		else: 
			return True

	sub_tree = {}

	# determine the entropy of each attribute
	for key, value in attributes.items():
		entropies[key] =  calc_entropy(training_data_frame, {key: value})

	# determine which attribute had the greatest entropy
	highest_value = 0.0
	highest_attribute = ''
	for key, value in entropies.items():
		if value > highest_value:
			highest_value = value
			highest_attribute = key

	# remove the attribute with the highest entropy from the dictionary of attributes
	attribute_values = attributes.pop(highest_attribute)

	# create a tree for each possible value of the selected attribute
	for value in attribute_values:
		sub_tree[value] = build_tree(copy.deepcopy(training_data_frame.loc[training_data_frame[highest_attribute] == value]), copy.deepcopy(attributes))
	
	# add default/none to the end of the subtree 
	if num_true > num_false:
		sub_tree['None'] = True
	else:
		sub_tree['None'] = False

	tree = (highest_attribute, sub_tree)
	return tree

# calc_entropy() function accepts a pandas dataframe and attribute dictionary as parameters
#	the the dataframe is all the test samples that you want to use to calculate entropy
#	the attribute dictionary should look like {'attribute' : ['possible', 'values']}
def calc_entropy(training_data_frame, attribute):
	total_num_records = len(training_data_frame.index)	# total number of test samples
	num_true = len(training_data_frame.loc[training_data_frame['class'] == True])	# number of test samples in class True
	num_false = len(training_data_frame.loc[training_data_frame['class'] == False])	# number of test samples in class False
	prob_false = float(num_false/total_num_records)	# probabilty a sample is false
	prob_true = float(num_true/total_num_records)	# probability a sample is true
	gain = 0	# initialize gain variable to 0

	# check to make sure prob_true and prob_false are not zero(or near to it) before summing the gain
	if prob_true > 0.00000000000001:
		gain -= (prob_true * np.log2(prob_true))
	if prob_false > 0.00000000000001:
		gain -= (prob_false * np.log2(prob_false))

	# in this case key = attribute and values = the array of possible values
	for key, values in attribute.items():
		# for each of the possible values sum the entropy calculations to gain as long as prob_true and prob_false aren't zero(or near to it)
		for value in values:
			num_records = len(training_data_frame.loc[training_data_frame[key] == value])	# this is an ugly looking way to find the number samples where the attribute = value

			# if there are no records continue the loop
			# (avoid dividing by zero)
			if num_records < 0.0000000000001: 
				continue

			num_true = len(training_data_frame.loc[(training_data_frame['class'] == True) & (training_data_frame[key] == value)])	# number of samples where attribute = value and class = True
			num_false = len(training_data_frame.loc[(training_data_frame['class'] == False) & (training_data_frame[key] == value)])	# number of samples where attribute = value and class = False
			prob_false = float(num_false/num_records) 	# probability of false
			prob_true = float(num_true/num_records)		# probability of true

			# more entropy calculations as long as important stuff doesn't equal zero
			sum = 0
			if prob_true > 0.00000000000001:
				sum += (prob_true * np.log2(prob_true))
			if prob_false > 0.00000000000001:
				sum += (prob_false * np.log2(prob_false))

			gain += float(num_records/total_num_records) * sum

	return gain

# Function which accepts an input data file and parses it into an
# acceptable format for the read_data() function.
def read_file_data(input_file_name):
	# File is read using pandas, header line included.
    input_data = pd.read_csv(input_file_name, header=0)	
    # Creates list with each line of file placed into a dictionary.
    data_list = list(input_data.T.to_dict().values())
    # Empty list for our final formatted data
    final_dict_list = []
    # For each dictionary row
    for data_row in data_list:
    	# Holds the last key in dictionary
    	last_key = list(data_row.keys())[-1]
    	# Holds the last value in dictionary
    	last_value = list(data_row.values())[-1]
    	# Removes the last 'key': value set from dictionary
    	data_row.pop(last_key)
    	# Creates a new tuple with dictionary, last value
    	new_tuple = (data_row, last_value)
    	# Appends new tuple to final List to fit the training data format
    	final_dict_list.append(new_tuple)
    # Sends newly constructed list to read_data () method
    read_data(final_dict_list)

# classify() function accepts a decision tree(like the one returned form build_tree()) and test sample(which is a dictionary)
#	and returns whether the sample belongs to class true or false
def classify(dt, sample):
	# loop through each of the samples attribute, value pairs
	for sample_attribute, sample_attribute_value in sample.items():
		# find the sample's attribute that matches the current tree node's attribute
		if sample_attribute == dt[0]:
			# iterate throught the dictionary of tree attribute values to trees/leaves
			for tree_attribute_value, sub_tree in dt[1].items():
				# if the tree value matches the sample value
				if tree_attribute_value ==  sample_attribute_value:
					# check if its a leaf node, if so return the value(victory)
					if type(sub_tree) == type(True):
						return sub_tree
					# otherwise continue the search on the next subtree
					else:
						sample.pop(sample_attribute)
						return classify(dt[1][sample_attribute_value], sample)
	# if none of the sample attributes match the one of the tree node
	# or none of the sample values match the ones in the tree node return the None/default case
	return dt[1]['None']		

if len(sys.argv) > 1:
    input_file_name = sys.argv[1]
    read_file_data(input_file_name)
else:
    read_data(training_data_input)