import cv2
import numpy as np

 #Pri implementaciji algoritmov je bil uporabljen Chat-GPT

def spremeni_kontrast(slika, alfa, beta):
    # Scale the pixel values based on the contrast and brightness
    spremenjena_slika = alfa * slika + beta
    # Clip the pixel values to ensure they stay within the valid range
    spremenjena_slika = np.clip(spremenjena_slika, 0, 255)
    # Convert the adjusted NumPy array back to a 2D NumPy array and return it
    spremenjena_slika = spremenjena_slika.astype(np.uint8)
    return spremenjena_slika
