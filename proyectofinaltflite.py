import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:\\Users\\Juan\\Documents\\ProyectoTinyLite\\haarcascade_frontalface_default.xml')
interpreter = tf.lite.Interpreter(model_path = 'C:\\Users\\Juan\\Documents\\ProyectoTinyLite\\my_model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

class_names=['Withmask', 'Withoutmask']
img=cv2.imread('C:\\Users\\Juan\\Documents\\ProyectoTinyLite\\opencv_frame_0.png')
img2=cv2.resize(img,(224,224))
img2 = np.reshape(img2,[1,224,224,3])
img2 = img2/255.0
test_imgs_numpy = np.array(img2, dtype=np.float32)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'],test_imgs_numpy )
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions)
prediction_classes = np.argmax(tflite_model_predictions, axis=1)
print(class_names[prediction_classes[0]])
prediccion = class_names[prediction_classes[0]]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    if prediccion == 'Withmask':
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
plt.title(prediccion)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
cv2.waitKey()
