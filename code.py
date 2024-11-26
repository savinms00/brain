import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog, ttk
from ttkthemes import ThemedTk
import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Define paths
image_directory = "C:\\mask project\\image_dataset\\images"
mask_directory = "C:\\mask project\\image_dataset\\masks"

# Load data
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

dataset = []
label = []
INPUT_SIZE = 64

# No tumor
for image_name in no_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Tumor
for image_name in yes_tumor_images:
    if image_name.lower().endswith('.jpg'):
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# Split data
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train model with data augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=16), 
          epochs=30, 
          validation_data=(x_test, y_test),
          class_weight={0: 1.0, 1: 2.0})  # Adjust class weights as needed

# Save model
model.save('BrainTumor_VGG16.h5')

# Function to speak the result
def speak_result(result_text):
    engine.say(result_text)
    engine.runAndWait()

# Function to calculate tumor percentage
def calculate_tumor_percentage(image, mask):
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # Resize mask to match the dimensions of the input image
    mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Calculate tumor area
    tumor_area = np.sum(mask_resized == 1)
    
    # Calculate total area
    total_area = image.shape[0] * image.shape[1]
    
    # Calculate tumor percentage
    tumor_percentage = (tumor_area / total_area) * 400  # Corrected percentage calculation
    
    return tumor_percentage

# Function to determine danger level based on tumor percentage
def determine_danger_level(tumor_percentage):
    if tumor_percentage < 7:
        return "Low Risk"
    elif tumor_percentage < 14:
        return "Moderate Risk"
    else:
        return "High Risk"

# Function to outline tumor region
def outline_tumor(image, mask):
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # Resize mask to match the dimensions of the input image
    mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a color mask for visualization
    tumor_only_image = np.zeros_like(image)
    tumor_only_image[mask_resized == 1] = image[mask_resized == 1]
    
    return tumor_only_image

# GUI for prediction
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print("Error loading image for prediction.")
            return
        
        # Determine the mask file path
        base_name = os.path.basename(file_path)
        mask_file_name = base_name.replace('.jpg', '_mask.png')
        mask_file_path = os.path.join(mask_directory, mask_file_name)
        
        # Load the mask as grayscale
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Error loading mask: {mask_file_path}")
            return
        
        # Predict using the model
        img = Image.fromarray(image)
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        img_array = np.array(img) / 255.0  # Ensure consistent normalization
        input_img = np.expand_dims(img_array, axis=0)
        result = model.predict(input_img)

        result_text = "Tumor Detected" if result[0][0] > 0.5 else "No Tumor Detected"
        
        # Speak only the detection result
        speak_result(result_text)
        
        # Update the GUI with the detection result
        if result[0][0] > 0.5:
            tumor_percentage = calculate_tumor_percentage(image, mask)
            danger_level = determine_danger_level(tumor_percentage)
            result_label.config(text=f"{result_text}\nTumor Percentage: {tumor_percentage:.2f}%\nDanger Level: {danger_level}")

            # Display the tumor-only image
            tumor_only_image = outline_tumor(image, mask)
            tumor_only_image = Image.fromarray(cv2.cvtColor(tumor_only_image, cv2.COLOR_BGR2RGB))
            tumor_only_image = tumor_only_image.resize((300, 300))
            tumor_only_img_tk = ImageTk.PhotoImage(tumor_only_image)
            outlined_img_label.config(image=tumor_only_img_tk)
            outlined_img_label.image = tumor_only_img_tk
        else:
            result_label.config(text=result_text)
            # Clear the tumor outline label if no tumor is detected
            outlined_img_label.config(image='')
            outlined_img_label.image = None

        # Display the original image
        original_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        original_image = original_image.resize((300, 300))
        original_img_tk = ImageTk.PhotoImage(original_image)
        original_img_label.config(image=original_img_tk)
        original_img_label.image = original_img_tk

# Create GUI
root = ThemedTk(theme="equilux")
root.title("Brain Tumor Detection")
root.geometry("900x900")

style = ttk.Style()
style.configure('TButton', font=('Arial', 14, 'bold'), padding=10, background='#00cc00', foreground='white')
style.map('TButton', background=[('active', '#00b300')])
style.configure('TLabel', font=('Arial', 16), padding=10, background='#2b2b2b', foreground='white')
root.configure(bg='#2b2b2b')
frame = ttk.Frame(root, padding="20", style='TFrame')
frame.pack(expand=True, fill='both')
style.configure('TFrame', background='#2b2b2b')

btn = ttk.Button(frame, text="Load Image", command=load_image, style='TButton')
btn.pack(pady=20)

original_img_label = ttk.Label(frame, style='TLabel')
original_img_label.pack(side='left', padx=10)

outlined_img_label = ttk.Label(frame, style='TLabel')
outlined_img_label.pack(side='right', padx=10)

result_label = ttk.Label(frame, text="", style='TLabel')
result_label.pack(pady=20)

root.mainloop()
