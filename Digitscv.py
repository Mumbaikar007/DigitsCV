
import cv2
import numpy as np      # for hsplit
from PIL import Image   # python-imaging library for resizing


def Centroid_x_coordinate ( contour ):


    if ( cv2.contourArea( contour ) > 10):

        M = cv2.moments( contour )
        return int ( M['m10'] / M['m00'])

def make_square ( image ):

    height = image.shape[0]
    width = image.shape[1]

    if ( height == width ):
        return image

    height = 2 * height
    width = 2 * width

    image = cv2.resize( image, (width, height), interpolation=cv2.INTER_CUBIC)

    if ( height > width ):

        pad = ( height - width ) / 2
        return cv2.copyMakeBorder( image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])

    pad = ( width - height ) / 2
    return cv2.copyMakeBorder( image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])



def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions

    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg



dataset = cv2.imread ('digits.png',0)

# cv2.pyrDown() and cv2.pyrUp() functions ...
# Higher level (Low resolution) in a Gaussian Pyramid is formed by removing consecutive rows and columns in Lower
# level (higher resolution) image. Then each pixel in higher level is formed by the contribution from 5 pixels in
# underlying level with gaussian weights. By doing so, a M x N image becomes M/2 x N/2 image. So area reduces
# to one-fourth of original area. It is called an Octave.
#small_dataset = cv2.pyrDown( dataset )

# numpy.vsplit - vertical split
# numpy.hsplit - horizontal split
# making a list data-type ( 50, 100, 20, 20) out of 5000 pixels dataset
dataset_4d_list = [np.hsplit(row, 100) for row in np.vsplit( dataset, 50)]

# making numpy array out of the list
dataset_numpy_array = np.array(dataset_4d_list)


# Split the full data set into two segments 70-30 spilt
# One will be used fro Training the model, the other as a test data set
train_data = dataset_numpy_array[:, :70].reshape(-1, 400).astype(np.float32)  # Size = (3500,400)
test_data = dataset_numpy_array[:, 70:100].reshape(-1, 400).astype(np.float32)  # Size = (1500,400)


# Making labels for ML
k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_labels = np.repeat(k, 350)[:, np.newaxis]
test_labels = np.repeat(k, 150)[:, np.newaxis]


# Machine Learning
kth_nearest_neighbour = cv2.KNearest()
kth_nearest_neighbour.train( train_data, train_labels)
ret, result, neighbors, distance = kth_nearest_neighbour.find_nearest( test_data, k=3)


# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
# matches = (result == test_labels)
# correct = np.count_nonzero(matches)
# accuracy = correct * (100.0 / result.size)
# print("Accuracy is = %.2f" % accuracy + "%")

image = cv2.imread( 'numbers.jpg')

# Turn image gray
image_gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)

# Blurring image
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# image thresholding
#ret_thresh, image_thresh = cv2.threshold( image_blurred, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#image_thresh = cv2.adaptiveThreshold(image_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                cv2.THRESH_BINARY,11,2)
#image_thresh = cv2.bitwise_not( image_thresh)
#cv2.imshow( 'thresh', image_thresh)

image_edged = cv2.Canny(image_blurred, 30, 70)


# finding numbers (contours) ; RETR_EXTERNAL - only the eldest contours
contours, hierarchy = cv2.findContours( image_edged.copy(),cv2.RETR_EXTERNAL, \
               cv2.CHAIN_APPROX_SIMPLE)

# Sort out contours left to right by using their x cordinates
contours = sorted(contours, key= Centroid_x_coordinate, reverse=False)
print(len(contours))

full_number = []

# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)

    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow("Contours", image)

    if w >= 5 and h >= 5:
        roi = image_blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        squared = make_square(roi)
        final = resize_to_pixel(20, squared)

        cv2.imshow( 'roi', final)
        final_array = final.reshape((1, 400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = kth_nearest_neighbour.find_nearest(final_array, k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, number, (x, y + 155),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)

#cv2.imshow( 'pyrdown', image_thresh)
print( full_number)
cv2.waitKey(0)
cv2.destroyAllWindows()




