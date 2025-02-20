from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to='uploads/')
    extracted_text = models.TextField(blank=True, null=True)
    translated_text = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} uploaded at {self.uploaded_at}"
    
    
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)