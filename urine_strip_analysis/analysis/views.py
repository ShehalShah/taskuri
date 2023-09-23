from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import UrineStripAnalysis
# from .serializers import UrineStripAnalysisSerializer
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import json

@api_view(['POST'])
def upload_image(request):
    if 'image' not in request.FILES:
        return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

    uploaded_image = request.FILES['image']

    try:
        # Perform image analysis and extract RGB values
        color_values = extract_colors(uploaded_image)

        # Create an instance of UrineStripAnalysis and set the colors
        analysis_instance = UrineStripAnalysis()
        analysis_instance.set_colors(color_values)

        # Save the instance
        analysis_instance.image = uploaded_image  # Save the uploaded image
        analysis_instance.save()

        return Response({"message": "Analysis saved successfully","result":color_values}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def extract_colors(uploaded_image):
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    num_pixels = image_array.shape[0] * image_array.shape[1]
    image_array_reshaped = image_array.reshape(num_pixels, -1)
    num_colors = 10
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_array_reshaped)
    colors = kmeans.cluster_centers_
    colors = colors.astype(int)
    result = {}
    color_names = ['URO', 'BIL', 'KET', 'BLD', 'PRO', 'NIT', 'LEU', 'GLU', 'SG', 'PH']
    
    for i, color in enumerate(colors):
        result[color_names[i]] = color.tolist()
    
    return result

# @api_view(['GET'])
# def get_analysis(request, analysis_id):
#     try:
#         analysis_instance = UrineStripAnalysis.objects.get(pk=analysis_id)
#         serializer = UrineStripAnalysisSerializer(analysis_instance)
#         return Response(serializer.data, status=status.HTTP_200_OK)
#     except UrineStripAnalysis.DoesNotExist:
#         return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)
