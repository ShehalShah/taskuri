from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import UrineStripAnalysis
# from .serializers import UrineStripAnalysisSerializer
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import json
import cv2
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile

@api_view(['POST'])
def upload_image(request):
    if 'image' not in request.FILES:
        return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

    uploaded_image = request.FILES['image']

    try:
        # # Perform image analysis and extract RGB values
        # color_values = extract_colors(uploaded_image)

        # # Create an instance of UrineStripAnalysis and set the colors
        # analysis_instance = UrineStripAnalysis()
        # analysis_instance.set_colors(color_values)

        # # Save the instance
        # analysis_instance.image = uploaded_image  # Save the uploaded image
        # analysis_instance.save()
        color_strip = find_color_strip(uploaded_image)

        # Extract the RGB values of the colors within the strip
        color_values = extract_colors(color_strip)

        return Response({"message": "Analysis saved successfully","result":color_values}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
# def extract_colors(image):
#     # Define the number of boxes to extract (10 boxes)
#     num_boxes = 10

#     # Get the dimensions of the image
#     image_height, image_width, _ = image.shape

#     # Initialize a dictionary to store the RGB values of each box
#     result = {}
#     color_names = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']

#     # Initialize variables for box detection
#     box_start_y = 35  # Starting y-coordinate for the first box
#     box_height = 80  # Expected box height
#     spacing = 60  # Expected spacing between boxes
#     color_threshold = 30  # Color shade threshold

#     # Loop to detect and extract each box
#     for i in range(num_boxes):
#         # Define the center position for the current box
#         center_x = image_width // 2
#         center_y = box_start_y + (i * (box_height + spacing))

#         # Get the color at the center of the current box
#         center_color = image[center_y, center_x]

#         # Define color ranges based on the center color
#         lower_color = center_color - color_threshold
#         upper_color = center_color + color_threshold

#         # Create a mask for the color range
#         mask = cv2.inRange(image, lower_color, upper_color)

#         # Find the first and last non-zero pixels along the vertical axis
#         nonzero_indices = np.where(mask > 0)
        
#         if len(nonzero_indices[0]) > 0:
#             top_bound = min(nonzero_indices[0])
#             bottom_bound = max(nonzero_indices[0]) + 1  # Adding 1 to include the last row
            
#             # Crop the current box based on the color detection
#             box = image[top_bound:bottom_bound, :]

#             # Calculate the average color in the current box
#             average_color = np.mean(box, axis=(0, 1)).astype(int)

#             # Store the average color in the result dictionary
#             result[color_names[i]] = average_color.tolist()
#         else:
#             # If no matching color found, add None to the result
#             result[color_names[i]] = None

#     return result


import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image):
    # Define the number of boxes you want to identify (in your case, 10)
    num_boxes = 10

    # Get the dimensions of the input image
    height, width, _ = image.shape

    # Calculate the approximate height of each box
    box_height = height // num_boxes

    # Initialize an empty list to store the colors of the boxes
    box_colors = []

    # Iterate through the image vertically to extract colors of each box
    for i in range(num_boxes):
        # Calculate the starting and ending rows for the current box
        start_row = i * box_height
        end_row = (i + 1) * box_height

        # Extract the region corresponding to the current box
        box_region = image[start_row:end_row, :]

        # Reshape the box region into a 1D array of RGB values
        box_array = box_region.reshape(-1, 3)

        # Perform K-means clustering on the box region
        kmeans = KMeans(n_clusters=1)  # Use 1 cluster to get the dominant color
        kmeans.fit(box_array)

        # Get the color of the cluster center
        box_color = kmeans.cluster_centers_[0].astype(int)

        # Append the color to the list of box colors
        box_colors.append(box_color.tolist())

    # Create a dictionary to associate box numbers with their colors
    result = {f'Box {i + 1}': color for i, color in enumerate(box_colors)}

    return result


def find_color_strip(uploaded_image):
    try:
        # Read the image data from the uploaded image
        image_data = BytesIO(uploaded_image.read())
        image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

        # Define color thresholds for identifying the vertical lines
        lower_color_threshold = np.array([0, 0, 0], dtype=np.uint8)  # Black or near black
        upper_color_threshold = np.array([50, 50, 50], dtype=np.uint8)  # Tolerate some variation

        # Create a mask to identify pixels within the color threshold
        mask = cv2.inRange(image, lower_color_threshold, upper_color_threshold)

        # Find the coordinates of the first and last non-zero pixel along the horizontal axis
        left_bound = np.argmax(mask[45:, :], axis=1).min()  # Start from row 45 (adjust if needed)
        right_bound = np.argmax(mask[45:, :], axis=1).max()  # Start from row 45 (adjust if needed)

        # Calculate the top and bottom coordinates of the strip
        top_bound = 45  # Adjust if needed
        bottom_bound = top_bound + 940  # Size of the strip

        # Crop the rectangular area based on the identified coordinates
        color_strip = image[top_bound:bottom_bound, right_bound-71:right_bound]

        # Save the cropped image for debugging purposes
        cv2.imwrite('color_strip.jpg', color_strip)

        return color_strip

    except Exception as e:
        raise Exception(str(e))



# def find_color_strip(image_path):
#     # Load the image
#     # image = cv2.imread(image_path)
#     # cv2.imshow(image_path)
#     image = np.array(Image.open(image_path))
#     # cv2.imwrite('color_strip.jpg', image)
#     # cv2.imshow('color_strip.jpg',image)
    

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding to segment the color strip
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Sort the contours by their y-coordinate (top to bottom)
#     sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

#     # Initialize variables to keep track of the color strip boundaries
#     top = None
#     bottom = None

#     # Iterate through the contours to identify the color strip
#     for contour in sorted_contours:
#         # Get the bounding box of the current contour
#         x, y, w, h = cv2.boundingRect(contour)

#         # Check if the width and height of the bounding box are similar (a square shape)
#         if 0.9 <= w / h <= 1.1:
#             if top is None:
#                 top = y
#             bottom = y + h

#     # Crop the region containing the color strip from the original image
#     color_strip = image[top:bottom, :]

#     # Save the color strip image for debugging purposes
#     # cv2.imwrite('color_strip.jpg', color_strip)

#     return color_strip


# def find_color_strip(image_path):
#     # Load the image
#     # image = cv2.imread(image_path)
#     image = np.array(Image.open(image_path))
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply thresholding to segment the color strip
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Iterate through the contours and find the largest one
#     largest_contour = max(contours, key=cv2.contourArea)
    
#     # Get the coordinates of the bounding box around the color strip
#     x, y, w, h = cv2.boundingRect(largest_contour)
    
#     # Crop the region containing the color strip
#     color_strip = image[y:y+h, x:x+w]
#     cv2.imwrite('color_strip.jpg', color_strip)
#     return color_strip

# def extract_colors(uploaded_image):
#     image = Image.open(uploaded_image)
#     image_array = np.array(image)
#     num_pixels = image_array.shape[0] * image_array.shape[1]
#     image_array_reshaped = image_array.reshape(num_pixels, -1)
#     num_colors = 10
#     kmeans = KMeans(n_clusters=num_colors)
#     kmeans.fit(image_array_reshaped)
#     colors = kmeans.cluster_centers_
#     colors = colors.astype(int)
#     result = {}
#     color_names = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    
#     for i, color in enumerate(colors):
#         result[color_names[i]] = color.tolist()
    
#     return result




# @api_view(['GET'])
# def get_analysis(request, analysis_id):
#     try:
#         analysis_instance = UrineStripAnalysis.objects.get(pk=analysis_id)
#         serializer = UrineStripAnalysisSerializer(analysis_instance)
#         return Response(serializer.data, status=status.HTTP_200_OK)
#     except UrineStripAnalysis.DoesNotExist:
#         return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)