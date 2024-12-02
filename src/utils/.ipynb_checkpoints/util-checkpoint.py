import base64
import uuid
import os


from .envs import sandbox_image_save_dir



def image_bytes_to_png(image):
    try:
        image_bytes = base64.b64decode(image)
        image_path = os.path.join(sandbox_image_save_dir, f"{str(uuid.uuid4())}.png")
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        return image_path
    except Exception as e:
        return f"Image Save Error: {repr(e)}" 
    
