import cv2

class ImageMerger(object):
	"""
	Class that handles merging multiple images together.

	"""

	def __init__(self):
		self.image_buffer = [] # Holds 2 images, a merged image and a new image
		self.image_counter = 0 # Counts number of images that have merged

	def merge(self, image, num_imgs_to_merge):
		"""
		Merges images together. 

		:param image: Image to load into buffer and merge with others
		:param num_imgs_to_merge: Number of images to merge together

		If you choose a num_imgs_to_merge of 3, for example, 
		output image 1 will be 100% of first image, image 2 will be
		50% first and second, image 3 will be 25% first, 25% second and
		50% third, image 4 will be 100% of the fourth and so on.

		"""

		# Add image to buffer and increase counter
		self.add_to_buffer(image)
		self.image_counter += 1

		# Return first image if there is only one in buffer
		if len(self.image_buffer) == 1:
			return self.image_buffer[0]

		elif len(self.image_buffer) == 2:
			# Else merge the two images in buffer together
			weight = 0.5 # Weight of first image in buffer
			merged_image = cv2.addWeighted(self.image_buffer[0], weight,
										   self.image_buffer[1], 1 - weight, 0)

			# Clear buffer
			self.clear_buffer()

			# If more images need to be merged, add last merged image to buffer
			if self.image_counter is not num_imgs_to_merge:
				self.add_to_buffer(merged_image)

			# Return merged image
			return merged_image

		else:
			# Should not get here
			raise ValueError("Error: buffer overflow")

	def add_to_buffer(self, image):
		self.image_buffer.append(image)

	def clear_buffer(self):
		self.image_buffer[:] = []
