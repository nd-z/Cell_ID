import numpy as np
import cv2

def filter_big_cells(image):
	#threshold blue ranges
	#BGR ordering
	lower_blue = np.array([90,80,160])
	upper_blue = np.array([150,140,255])

	mask = cv2.inRange(image, lower_blue, upper_blue)
	return mask

def filter_small_cells(image):
	#threshold pink/red ranges
	#BGR ordering
	lower_pink = np.array([140,57,59])
	upper_pink = np.array([165,115,185])

	mask = cv2.inRange(image, lower_pink, upper_pink)
	return mask

def find_big_cells(contours, big_bound):
	big_cells = []

	for c in contours:
		c_area = cv2.contourArea(c)

		if c_area > big_bound:
			big_cells.append(c)

	return big_cells

def find_small_cells(contours, small_bound):
	small_cells = []

	for c in contours:
		c_area = cv2.contourArea(c)

		if c_area > small_bound:
			small_cells.append(c)

	return small_cells

def classify_BLM(cnt, cnt_approx, check):
	if check == True:
		return 'LYMPHOCYTE'
	else:
		# TODO fine-tune distinction btwn B/M

		points = cv2.convexHull(cnt_approx, 5, False)

		# ideally, train on the convexity defects to more accurately distinguish between Monocytes and Basophils
		if(len(points) > 4):
			return 'MONOCYTE'
		else:
			return 'BASOPHIL'

def run(image_path):
	path = './images/' + image_path
	image = cv2.imread(path)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# isolate all the big cells (bright blue)
	mask = filter_big_cells(hsv)


	# extract contours of big cells
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# filter out any small contours that are detected after thresholding
	# ideally, also train this bound number
	big_bound = 650
	big_cells = find_big_cells(contours, big_bound)

	# Extract contours of small cells
	mask = filter_small_cells(hsv)

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	small_bound = 400
	small_cells = find_small_cells(contours, small_bound)

	# classify
	ret_type = ''
	if len(small_cells) > 0 and len(big_cells) <= 2:
		ret_type = 'EOSINOPHIL'
	elif len(big_cells) > 1 and len(small_cells) == 0:
		ret_type = 'NEUTROPHIL'
	elif len(small_cells) == 0:

		# the hard part: distinguishing between Lymphocytes, Monocytes, and Basophils

		# best approach is probably analyzing/training on contours; training is not used here, but ideally i would train on the convexity defects

		cnt = big_cells[0]

		# cnt_approx should be close to cnt; goal is to smooth the contours for easier detection of Lymphocytes, while also making sure Basophils and Monocytes are detectable with a convexity check
		epsilon = 0.03 * cv2.arcLength(cnt,True)
		cnt_approx = cv2.approxPolyDP(cnt,epsilon,True)

		# check is really just for Lymphocytes
		check = cv2.isContourConvex(cnt_approx)

		ret_type = classify_BLM(cnt, cnt_approx, check)

	return ret_type