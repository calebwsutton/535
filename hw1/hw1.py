# CSC 535 Homework 1 Part 1

import pandas as pd
import numpy as np
import copy

training_data = [
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

def main():
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

	print(build_tree(training_data_frame, attributes))

def build_tree(training_data_frame, attributes):
	num_true = 0
	num_false = 0
	entropies = {}
	tree = ()


	# count number of values in each class
	for value in training_data_frame['class']:
		if value == true:
			num_true += 1
		else 
			num_false +=1

	# if all values of class are true or false return that value
	if num_true == 0:
		return False
	else if num_false == 0:
		return True

	# if there are no attributes left return the most common class
	if len(attributes) == 0:
		if num_false > num_true:
			return False
		else: 
			return True

	sub_tree = {}

	# #if there is only on attribute left	
	# if len(attributes) == 1:
		
	# 	#Split on the last attribute
	# 	for key, values in attributes.items():
	# 		attributes.popitem()
	# 		for value in values:
	# 			sub_tree[value] = build_tree(copy.deepcopy(training_data_frame.loc[training_data_frame[key] == value]), copy.deepcopy(attributes))
				
	# 		tree = (key, sub_tree)
	# 	#print(tree)
	# 	# add None/Default value to end of subtree	
	# 	return tree


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
		sub_tree['default'] = True
	else:
		sub_tree['default'] = False


	tree = (highest_attribute, sub_tree)
	return tree



def calc_entropy(training_data_frame, attribute):
	total_num_records = len(training_data_frame.index)
	num_true = len(training_data_frame.loc[training_data_frame['hire'] == True])
	num_false = len(training_data_frame.loc[training_data_frame['hire'] == False])
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

def classify(sample):
	return True

main()