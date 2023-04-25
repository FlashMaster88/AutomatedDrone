import numpy as np
import cv2
from imutils.video import FPS

class Detector:
	def __init__(self, use_cuda = False):

		#read model
		self.faceModel = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt", caffeModel= "models/MobileNetSSD_deploy.caffemodel")

		if use_cuda:
			self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	def processImage(self, imgName):
		#self.img = cv2.imread(imgName)
		self.img = imgName
		
		(self.height, self.width) = self.img.shape[:2]

		self.processFrame()
		
		cv2.imshow("Output", self.img)
		cv2.waitKey(1)
	
	def processVideo(self, videoName):
		cap = cv2.VideoCapture(videoName)
		
		if (cap.isOpened() == False):
			print("Error opening video...")
			return
		
		(sucess, self.img) = cap.read()
		(self.height, self.width) = self.img.shape[:2]
		
		print(type(cap))
		print(type(self.img))
		fps = FPS().start()

		while sucess:
			self.processFrame()
			cv2.imshow("output", self.img)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

			fps.update()
			(sucess, self.img) = cap.read()

		fps.stop()
		print("Elapsed time: {:.2f}".format(fps.elapsed()))
		print("FPS: {:.2f}".format(fps.fps()))

		cap.release()
		cv2.destroyAllWindows()
	
	def processFrame(self):
		height, width = self.img.shape[0], self.img.shape[1]
		classes = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

		np.random.seed(543218)
		colors = np.random.uniform(0, 255, size=(len(classes), 3))
		blob = cv2.dnn.blobFromImage(cv2.resize(self.img, (300, 300)), 0.007, (300, 300), 130)
		self.faceModel.setInput(blob)
		predictions = self.faceModel.forward()

		print(predictions[0][0][0])
		print(predictions[0][0][1])
		print(predictions[0][0][2])
		
		for i in range(0, predictions.shape[2]):
			confidence = predictions[0][0][i][2]

			upper_left_x = int(predictions[0, 0, i, 3] * width)
			upper_left_y = int(predictions[0, 0, i, 4] * height)
			lower_right_x = int(predictions[0, 0, i, 5] * width)
			lower_right_y = int(predictions[0, 0, i, 6] * height)

			if predictions[0, 0, i, 2] > 0.5:
				class_index = int(predictions[0, 0, i, 1])
				bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
				(xmin, ymin, xmax, ymax) = bbox.astype("int")
				prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
				
				cv2.rectangle(self.img, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 2)
				cv2.putText(self.img, prediction_text, (upper_left_x, upper_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

				print('upper_left_x: ' + str(upper_left_x))
				print('upper_left_y: ' + str(upper_left_y))
				print('lower_right_x: ' + str(lower_right_x))
				print('lower_right_y: ' + str(lower_right_y))
