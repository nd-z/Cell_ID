import numpy as np
import cv2
import classifier
import os, sys

'''
Runs the classifier on a dataset and keeps track of false positives/negatives for each class
'''

e_fp = 0
e_fn = 0
n_fp = 0
n_fn = 0
b_fp = 0
b_fn = 0
l_fp = 0
l_fn = 0
m_fp = 0
m_fn = 0

def add_fp(name):
	global e_fp, n_fp, b_fp, l_fp, m_fp

	if name == 'EOSINOPHIL':
		e_fp = e_fp + 1
	elif name == 'NEUTROPHIL':
		n_fp = n_fp + 1
	elif name == 'LYMPHOCYTE':
		l_fp = l_fp + 1
	elif name == 'BASOPHIL':
		b_fp = b_fp + 1
	elif name == 'MONOCYTE':
		m_fp = m_fp + 1

def add_fn(name):
	global e_fn, n_fn, b_fn, l_fn, m_fn

	if name == 'EOSINOPHIL':
		e_fn = e_fn + 1
	elif name == 'NEUTROPHIL':
		n_fn = n_fn + 1
	elif name == 'LYMPHOCYTE':
		l_fn = l_fn + 1
	elif name == 'BASOPHIL':
		b_fn = b_fn + 1
	elif name == 'MONOCYTE':
		m_fn = m_fn + 1

classified = []

with open('./output', 'w') as writer:
	for image_path in os.listdir('./images'):
		#print image_path
		name = classifier.run(image_path)
		writer.write(image_path + ', ' + name + '\n')
		#print(name)
		classified.append(name)

with open('./cell_classes') as file:
    lines = file.readlines()

lines = [l.strip() for l in lines] 

correct = 0
total = 0

i = 0

for line in lines:
	line = line.rstrip()
	if line == '':
		continue

	if line != classified[i]:
		add_fp(classified[i])
		add_fn(line)
	else:
		correct = correct + 1

	i = i + 1
	total = total + 1

print('Accuracy: ' + `correct` + '/' + `total`)
print('EOSINOPHIL False P/N: ' + `e_fp` + '/' + `e_fn`)
print('NEUTROPHIL False P/N: ' + `n_fp` + '/' + `n_fn`)
print('LYMPHOCYTE False P/N: ' + `l_fp` + '/' + `l_fn`)
print('BASOPHIL False P/N: ' + `b_fp` + '/' + `b_fn`)
print('MONOCYTE False P/N: ' + `m_fp` + '/' + `m_fn`)