import cv2 

# Function to extract frames 
def FrameCapture(path): 

	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Save every 10th frame with frame-count 
		if count % 2 == 0 and success:
			cv2.imwrite("C:/Users/Phalguni/Desktop/phalgun/Images/frame%d.jpg" % count, image) 
			# storing frames after extracting them from the video

		count += 1
	vidObj.release()

# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	FrameCapture("C:/Users/Phalguni/Desktop/phalgun/videos/video2.mp4")