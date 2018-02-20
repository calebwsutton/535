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

test_sample1 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"}
test_sample2 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"} 


def read_data(training_data):
	training_data_new = []
	attributes = {}

	for key, _ in training_data[0][0].items():
		attributes[key] = []

	for record in training_data:
		new_record = record[0]
		new_record['class'] = record[1]
		training_data_new.append(new_record)

	training_data_frame = pd.DataFrame(training_data_new)

	for key, _ in attributes.items():
		values = []
		for value in training_data_frame[key].unique():
			values.append(value)
		attributes[key] = values

	final_dt = (build_tree(training_data_frame, attributes))
	print (final_dt)
	print ("\n")
	print(classify(final_dt, test_sample1))
	print (classify(final_dt, test_sample2))
	# print (classify_input)

def build_tree(training_data_frame, attributes):
	num_true = 0
	num_false = 0
	entropies = {}
	tree = ()

	# count number of values in each class
	for value in training_data_frame['class']:
		if value == True:
			num_true += 1
		else :
			num_false +=1

	# if all values of class are true or false return that value
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

	#If there is more than 1 attribute left calculate the entropy for each attribute 
	for key, value in attributes.items():
		entropies[key] =  calc_entropy(training_data_frame, {key: value})

	#split on the highest entropy
	highest_value = 0.0
	highest_attribute = ''

	for key, value in entropies.items():
		if value > highest_value:
			highest_value = value
			highest_attribute = key

	attribute_values = attributes.pop(highest_attribute)
	for value in attribute_values:
		sub_tree[value] = build_tree(copy.deepcopy(training_data_frame.loc[training_data_frame[highest_attribute] == value]), copy.deepcopy(attributes))
	
	# add default to end of subtree 
	if num_true > num_false:
		sub_tree['None'] = True
	else:
		sub_tree['None'] = False

	tree = (highest_attribute, sub_tree)
	return tree


def calc_entropy(training_data_frame, attribute):
	total_num_records = len(training_data_frame.index)
	num_true = len(training_data_frame.loc[training_data_frame['class'] == True])
	num_false = len(training_data_frame.loc[training_data_frame['class'] == False])
	prob_false = float(num_false/total_num_records)
	prob_true = float(num_true/total_num_records)
	gain = 0

	if prob_true > 0.00000000000001:
		gain -= (prob_true * np.log2(prob_true))
	if prob_false > 0.00000000000001:
		gain -= (prob_false * np.log2(prob_false))

	for key, values in attribute.items():
		for value in values:
			# num_records is the number of 
			num_records = len(training_data_frame.loc[training_data_frame[key] == value])
			if num_records < 0.0000000000001:
				continue
			num_true = len(training_data_frame.loc[(training_data_frame['class'] == True) & (training_data_frame[key] == value)])
			num_false = len(training_data_frame.loc[(training_data_frame['class'] == False) & (training_data_frame[key] == value)])
			prob_false = float(num_false/num_records)
			prob_true = float(num_true/num_records)

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

def classify(dt, sample):
	print (sample)
	first_level = dt[0]
	second_level =  dt[1]
	for key, value in sample.items():
		if key == first_level:
			value_tuple = second_level.get(value)
			next_attribute = value_tuple[0]
			for key, value in sample.items():
				if key == next_attribute:
					next_values = value_tuple[1]
					final_value = next_values.get(value)	
					return (final_value)
					

if len(sys.argv) > 1:
    input_file_name = sys.argv[1]
    read_file_data(input_file_name)
else:
    read_data(training_data_input)