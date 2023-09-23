from django.db import models
import json

class UrineStripAnalysis(models.Model):
    colors = models.JSONField(default=dict)  # Store RGB values as a dictionary
    image = models.ImageField(upload_to='urine_strips/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def set_colors(self, color_dict):
        self.colors = json.dumps(color_dict)

    def get_colors(self):
        return json.loads(self.colors)
