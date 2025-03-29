import cv2
import numpy as np
import os

def extract_logo(image_path, output_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a mask
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding box of the largest contour (assuming it's the logo)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        logo = image[y:y+h, x:x+w]
        
        # Save extracted logo
        cv2.imwrite(output_path, logo)
        print(f"Logo saved at {output_path}")
        
        # Show the extracted logo
        cv2.imshow("Extracted Logo", logo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No logo detected.")

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "blue logo.png")  # Change filename if needed
output_path = os.path.join(script_dir, "output_logo.png")

# Run the function
extract_logo(image_path, output_path)
