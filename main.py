import sys
import dlib
from skimage import io
#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
# file_name = "ImageDataset/ilham.jpeg"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

# Take the image file name from the command line
# file_name = "ImageFound/found.jpg"

# Load the image
# image = io.imread(file_name)

# Run the HOG face detector on the image data
# detected_faces = face_detector(image, 1)

# print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Show the desktop window with the image
# win.set_image(image)

# Loop through each face we found in the image
# for i, face_rect in enumerate(detected_faces):
#     # Detected faces are returned as an object with the coordinates
#     # of the top, left, right and bottom edges
#     print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
#                                                                              face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    # win.add_overlay(face_rect)

    # Get the the face's pose
    # pose_landmarks = face_pose_predictor(image, face_rect)
    #
    # # Draw the face landmarks on the screen.
    # win.add_overlay(pose_landmarks)
    #

#reading the image
img = imread('ImageFound/iswan.jpeg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

#resizing image
resized_img = resize(img, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
plt.show()
print(resized_img.shape)

#creating hog features
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image.shape)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()

# save the images
plt.imsave("resized_img.jpg", resized_img)
plt.imsave("hog_image.jpg", hog_image, cmap="gray")


dlib.hit_enter_to_continue()