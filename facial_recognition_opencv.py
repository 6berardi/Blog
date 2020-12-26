# You can find the blog post about this code at: www.gabriel-berardi.com/post/simple-facial-recognition-with-opencv

import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("people.jpeg")
gray_img = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, "gray")
plt.axis('off')
plt.show()

classifier_path = "~/haarcascade_frontalface_alt_tree.xml"
classifier = cv.CascadeClassifier(classifier_path)
faces = classifier.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=3)
print(faces)

c = img.copy()

for face in faces:
    x, y, w, h = face
    cv.rectangle(c, (x, y), (x+w, y+h), (0, 255, 0), 10)

plt.figure(figsize=(16,16))
img = cv.cvtColor(c, cv.COLOR_BGR2RGB)
plt.annotate(f'Number of detected faces: {len(faces)}', xy=(0.99, 0.02), xycoords='axes fraction',
             fontsize=20, color='green', bbox=dict(facecolor='black', alpha=0.99),
             horizontalalignment='right', verticalalignment='bottom')
plt.axis('off')
plt.imshow(img)
