# Image Inpainting & Generation with OpenAI DALL·E

This module provides a convenient Python interface for generating and editing images using OpenAI's DALL·E models. It supports:

- Generating new images from text prompts
- Inpainting selected regions of an existing image
- Automatic resizing/padding/stretching of images and masks
- Flask support for web-based image editing

## Requirements

```bash
pip install noahs_image_generation
```

Make sure your environment variable `OPENAI_API_KEY` is set, or pass your key directly into the class.

## Features

- Resize and pad images while maintaining aspect ratio
- Stretch images to exact dimensions
- Generate transparent masks from selected polygon regions
- Inpaint images with DALL·E 2
- Generate new images with DALL·E 3
- In-memory and file-based support

---

## Usage

### Initialization

```python
from noahs_image_generation import ImageGenerator

generator = ImageGenerator(api_key="your-openai-api-key")
```

---

### Generate an Image In-Memory

```python
prompt = "a cat astronaut riding a bicycle on Mars"
image_bytes = generator.generate_image_in_memory(prompt)

# Save or display the image
with open("cat_astronaut.png", "wb") as f:
    f.write(image_bytes.read())
```

---

### Inpaint an Image with a Mask (From File)

```python
output_bytes, mimetype = generator.inpaint(
    image_input="frog.png",
    coordinates=[(420, 25), (620, 25), (620, 300), (420, 300)], # rectangle selection on frog.png where you want the hat
    prompt="Give the frog a cowboy hat"
)

# Save the inpainted image
with open("frog_with_hat.png", "wb") as f:
    f.write(output_bytes.read())
```

### Inpaint an Image (From Bytes)

```python
# Read image into bytes
with open("frog.png", "rb") as f:
    image_bytes = f.read()

output_bytes, mimetype = generator.inpaint(
    image_input=image_bytes,
    coordinates=[(420, 25), (620, 25), (620, 300), (420, 300)],
    prompt="Give the frog a cowboy hat"
)

with open("frog_with_hat_from_bytes.png", "wb") as f:
    f.write(output_bytes.read())
```

---

### Inpaint an Image from a Flask Request

```python
# Assuming you are in a Flask route
@app.route("/inpaint", methods=["POST"])
def inpaint_route():
    generator = ImageGenerator()
    output_bytes, mimetype = generator.inpaint(flask_request=request)
    return send_file(output_bytes, mimetype=mimetype)
```

`curl` example to call the Flask endpoint:

```bash
curl -X POST http://localhost:5000/inpaint \
  -F "image=@my_image.png" \
  -F "coordinates=[[100,100],[200,100],[200,200],[100,200]]" \
  -F "prompt=draw a dragon tattoo on the arm"
```

---



