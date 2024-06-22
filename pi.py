import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="waste_classifier_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0  # Normalize to [0, 1]
    return img

# Function to classify the waste
def classify_waste(img):
    img = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)  # Returns the class index

# Servo control functions
servo_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz
pwm.start(0)

def set_servo_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

def papersort():
    print("Sorting paper...")
    set_servo_angle(90)  # Adjust the angle as needed

def plasticsort():
    print("Sorting plastic...")
    set_servo_angle(0)  # Adjust the angle as needed

def othersort():
    print("Sorting other waste...")
    set_servo_angle(180)  # Adjust the angle as needed

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Classify the current frame
        class_index = classify_waste(frame)
        print("Class Index:", class_index)

        # Control the servo based on the classification
        if class_index == 0:
            papersort()  # Paper
        elif class_index == 1:
            plasticsort()  # Plastic
        else:
            othersort()  # Other

finally:
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()