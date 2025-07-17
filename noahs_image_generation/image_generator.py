from openai import OpenAI
from PIL import Image
import requests
import io
import os
import cv2
import numpy as np
import json



def resize_and_pad(image_path, output_path, target_size=(1024, 1024), is_mask=False):
	"""
	Resizes an image to fit within a target size while maintaining aspect ratio
	and adding padding to make it a square.

	Args:
		image_path (str): Path to the input image.
		output_path (str): Path to save the resized and padded image.
		target_size (tuple): Target (width, height), default is (1024, 1024).
		is_mask (bool): If True, uses black padding (for masks). Otherwise, transparent.
	"""
	# Open the image
	img = Image.open(image_path)

	# Resize while maintaining aspect ratio
	img.thumbnail((target_size[0], target_size[1]), Image.LANCZOS)

	# Create a new blank image with the target size
	if is_mask:
		new_img = Image.new("L", target_size, 0)  # Black background for masks
	else:
		new_img = Image.new("RGBA", target_size, (0, 0, 0, 0))  # Transparent background for source images

	# Calculate center position for pasting
	paste_x = (target_size[0] - img.size[0]) // 2
	paste_y = (target_size[1] - img.size[1]) // 2

	# Paste resized image onto the blank image
	new_img.paste(img, (paste_x, paste_y))

	# Save the new image
	new_img.save(output_path)
	print(f"Saved resized image: {output_path}")




def stretch_image(image_path, output_path, target_size=(1024, 1024), is_mask=False):
	"""
	Stretches an image to exactly match the target dimensions.
	If is_mask=True, the output is converted to black and white (grayscale).

	Args:
		image_path (str): Path to the input image.
		output_path (str): Path to save the stretched image.
		target_size (tuple): Target (width, height), default is (1024, 1024).
		is_mask (bool): If True, converts the image to black and white (grayscale).
	"""
	# Open the image
	img = Image.open(image_path)

	# Resize the image (stretching to fit the new size)
	stretched_img = img.resize(target_size, Image.LANCZOS)

	# If it's a mask, convert to black & white (grayscale)
	if is_mask:
		stretched_img = stretched_img.convert("L")  # Convert to grayscale (0-255)

	# Save the new stretched image
	stretched_img.save(output_path)
	print(f"Saved {'mask' if is_mask else 'image'}: {output_path}")







def generate_mask_in_memory(image_io, coordinates):
	"""
	Creates a grayscale mask image with a specified editable region and converts it to a transparent PNG.
	
	Args:
		image_io (BytesIO): Input image as a BytesIO object.
		coordinates (tuple): Four (x, y) coordinates defining a rectangular region.
	
	Returns:
		BytesIO: Transparent mask image as a PNG in memory.
	"""
	image = Image.open(image_io).convert('RGBA')
	image_np = np.array(image)
	height, width = image_np.shape[:2]
	
	# Create a black mask (all pixels initially set to 0)
	mask = np.zeros((height, width), dtype=np.uint8)
	
	# Define the editable region (white area)
	(x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates
	pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
	cv2.fillPoly(mask, [pts], 255)  # Fill the selected area with white (editable)
	
	# Convert the grayscale mask to an RGBA image with transparency
	transparent_mask = Image.new("RGBA", (width, height))
	for x in range(width):
		for y in range(height):
			pixel = mask[y, x]
			alpha = 0 if pixel >= 128 else 255  # White -> Transparent, Black -> Opaque
			transparent_mask.putpixel((x, y), (0, 0, 0, alpha))
	
	# Save to a BytesIO object
	img_io = io.BytesIO()
	transparent_mask.save(img_io, format='PNG')
	img_io.seek(0)
	return img_io





def generate_mask(image_path, output_path, coordinates):
	"""
	Creates a grayscale mask image with a specified editable region and converts it to a transparent PNG.
	
	Args:
		image_path (str): Path to the source image (must be PNG).
		output_path (str): Path to save the final transparent mask.
		coordinates (tuple): Four (x, y) coordinates defining a rectangular region.
	"""
	# Load the source image to get dimensions
	image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	height, width = image.shape[:2]

	# Create a black mask (all pixels initially set to 0)
	mask = np.zeros((height, width), dtype=np.uint8)

	# Define the editable region (white area)
	(x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates
	pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
	cv2.fillPoly(mask, [pts], 255)  # Fill the selected area with white (editable)

	# Convert the grayscale mask to an RGBA image with transparency
	transparent_mask = Image.new("RGBA", (width, height))
	for x in range(width):
		for y in range(height):
			pixel = mask[y, x]
			alpha = 0 if pixel >= 128 else 255  # White -> Transparent, Black -> Opaque
			transparent_mask.putpixel((x, y), (0, 0, 0, alpha))

	# Save the transparent mask as a PNG
	transparent_mask.save(output_path, "PNG")
	print(f"Transparent mask saved as: {output_path}")















class ImageGenerator:
	def __init__(self, api_key=None):
		"""
		Initializes the DalleImageEditor with an OpenAI client.

		Args:
			api_key: Optional OpenAI API key. If not provided, will use the OPENAI_API_KEY environment variable.
		"""
		self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

	def inpaint_image(self, image_path, mask_path, prompt, output_path="inpainted_image.png"):
		"""
		Inpaints an image using a mask and a prompt, and saves the result to a file.

		Args:
			image_path: Path to the original image.
			mask_path: Path to the transparent mask image.
			prompt: Prompt to guide inpainting.
			output_path: File path to save the inpainted image.
		"""
		try:
			response = self.client.images.edit(
				model="dall-e-2",
				image=open(image_path, "rb"),
				mask=open(mask_path, "rb"),
				prompt=prompt,
				n=1,
				size="1024x1024"
			)
			image_url = response.data[0].url
			inpainted_image = Image.open(requests.get(image_url, stream=True).raw)
			inpainted_image.save(output_path)
			print(f"Inpainted image saved to {output_path}")
		except Exception as e:
			print(f"An error occurred during inpainting: {e}")

	def inpaint_image_in_memory(self, image_bytes, mask_bytes, prompt):
		"""
		Inpaints an image using a mask and a prompt and returns the image URL.

		Args:
			image_bytes: BytesIO of the original image.
			mask_bytes: BytesIO of the mask image.
			prompt: Prompt to guide inpainting.

		Returns:
			URL of the generated image.
		"""
		try:
			image_bytes.seek(0)
			mask_bytes.seek(0)
			image_bytes.name = "image.png"
			mask_bytes.name = "mask.png"

			response = self.client.images.edit(
				model="dall-e-2",
				image=image_bytes,
				mask=mask_bytes,
				prompt=prompt,
				n=1,
				size="1024x1024"
			)
			return response.data[0].url
		except Exception as e:
			print(f"An error occurred during inpainting in memory: {e}")
			return None

	def generate_image_to_file(self, prompt, output_path="generated_image.png", size="1024x1024"):
		"""
		Generates an image from a prompt and saves it to disk.

		Args:
			prompt: Description of the image to generate.
			output_path: File path to save the generated image.
			size: Size of the generated image (e.g., "1024x1024").

		Returns:
			File path to the saved image.
		"""
		try:
			response = self.client.images.generate(
				model="dall-e-3",
				prompt=prompt,
				n=1,
				size=size
			)
			image_url = response.data[0].url
			image = Image.open(requests.get(image_url, stream=True).raw)
			image.save(output_path)
			print(f"Generated image saved to {output_path}")
			return output_path
		except Exception as e:
			print(f"An error occurred during image generation: {e}")
			return None

	def generate_image_in_memory(self, prompt, size="1024x1024"):
		"""
		Generates an image from a prompt and returns it in memory.

		Args:
			prompt: Description of the image to generate.
			size: Size of the generated image.

		Returns:
			BytesIO object of the image.
		"""
		try:
			response = self.client.images.generate(
				model="dall-e-3",
				prompt=prompt,
				n=1,
				size=size
			)
			image_url = response.data[0].url
			image_bytes = io.BytesIO()
			image = Image.open(requests.get(image_url, stream=True).raw)
			image.save(image_bytes, format="PNG")
			image_bytes.seek(0)
			return image_bytes
		except Exception as e:
			print(f"An error occurred during in-memory image generation: {e}")
			return None


	def inpaint(self, image_input=None, coordinates=None, prompt=None, flask_request=None):
		"""
		Unified inpainting pipeline that supports multiple input formats.

		Args:
			image_input (Union[str, bytes, None], optional): Path to image or image bytes.
			coordinates (List[Tuple[int, int]], optional): List of 4 (x, y) tuples.
			prompt (str, optional): Inpainting prompt.
			flask_request (flask.Request, optional): Flask request object.

		Returns:
			(BytesIO, str): Tuple of PNG image bytes and MIME type.
		"""
		try:
			# --- CASE 1: Flask-based input ---
			if flask_request is not None:
				if 'image' not in flask_request.files:
					raise ValueError("No image file provided in request")

				image_file = flask_request.files['image']
				image = Image.open(image_file.stream).convert("RGBA")

				coordinates_str = flask_request.form.get("coordinates")
				prompt = flask_request.form.get("prompt")

				if not coordinates_str or not prompt:
					raise ValueError("Missing coordinates or prompt in form data")

				try:
					coordinates = json.loads(coordinates_str)
				except json.JSONDecodeError:
					raise ValueError("Invalid JSON format for coordinates")

			# --- CASE 2: Path-based input ---
			elif isinstance(image_input, str):
				image = Image.open(image_input).convert("RGBA")
				if not coordinates or not prompt:
					raise ValueError("coordinates and prompt are required with image path input")

			# --- CASE 3: Byte-based input ---
			elif isinstance(image_input, bytes):
				image = Image.open(io.BytesIO(image_input)).convert("RGBA")
				if not coordinates or not prompt:
					raise ValueError("coordinates and prompt are required with image bytes input")

			else:
				raise ValueError("Provide either a valid image_input or flask_request")

			# --- Prepare image in memory ---
			image_png_bytes = io.BytesIO()
			image.save(image_png_bytes, format="PNG")
			image_png_bytes.seek(0)
			image_png_bytes.name = "image.png"

			# --- Generate mask ---
			mask_bytes = generate_mask_in_memory(image_png_bytes, coordinates)
			mask_bytes.name = "mask.png"

			# --- Inpaint ---
			image_url = self.inpaint_image_in_memory(image_png_bytes, mask_bytes, prompt)

			if not image_url:
				raise RuntimeError("Failed to generate inpainted image")

			response = requests.get(image_url, stream=True)
			if response.status_code != 200:
				raise RuntimeError("Failed to download inpainted image")

			inpainted_image = Image.open(response.raw)
			output_bytes = io.BytesIO()
			inpainted_image.save(output_bytes, format="PNG")
			output_bytes.seek(0)
			return output_bytes, "image/png"

		except Exception as e:
			print(f"[ImageGenerator.inpaint] Error: {e}")
			raise e










if __name__ == "__main__":
	# flask usage
	# generator = ImageGenerator()
	# output_bytes, mimetype = generator.inpaint(flask_request=request)
	# request would need to look like this ...
	# curl -X POST http://localhost:5000/inpaint \
 #  -F "image=@input.png" \
 #  -F "coordinates=[[100,100],[200,100],[200,200],[100,200]]" \
 #  -F "prompt=add a top hat"



	# my_openai_api_key = "YOUR KEY HERE"
	# generator = ImageGenerator(api_key=my_openai_api_key)
	# output_bytes, mimetype = generator.inpaint(
	# 	image_input="shpwedms.png",
	# 	coordinates=[(420, 25), (620, 25), (620, 300), (420, 300)],
	# 	prompt="Give the frog a cowboy hat"
	# )
	# with open("result.png", "wb") as f:
	# 	f.write(output_bytes.read())

	# # image can also be passed in as bytes




	# my_openai_api_key = "YOUR KEY HERE"
	# generator = ImageGenerator(api_key=my_openai_api_key)
	# prompt = "realistic dumpy tree frog dressed as a cowboy setting on a can of baked beans"
	# generator.generate_image_to_file(prompt=prompt, output_path="generated_image.png", size="1024x1024")





