# PART 1: Imports and Setup
import uuid
import os
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai_text
from google import genai as genai_image
from google.genai import types
from cloud_upload import upload_image_local  # your Cloudinary upload function

# PART 2: Load environment variables and configure APIs
load_dotenv()
TEXT_API_KEY = os.getenv("GEMINI_TEXT_API_KEY")
IMAGE_API_KEY = os.getenv("GEMINI_IMAGE_API_KEY")

# Configure Gemini text API for prompt enhancement
genai_text.configure(api_key=TEXT_API_KEY)

# PART 3: Load BLIP model for image captioning (sketch to text)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=True  # Enables fast processor
)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# PART 4: Class for enhancing prompts using Gemini
class GeminiChatBot:
    """Uses Gemini text model to enhance/optimize prompts for image generation."""
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or (
            "You are a professional prompt optimizer for image generation. "
            "Transform user requests into rich, vivid, detailed, efficient prompts."
        )
        self.model = genai_text.GenerativeModel(
            model_name="gemini-1.5-flash-8b",
            system_instruction=self.system_prompt.strip(),
            generation_config={
                "temperature": 0.85,
                "top_p": 1,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
        )
        self.chat = self.model.start_chat(history=[])

    def enhance_prompt(self, user_prompt: str) -> str:
        try:
            response = self.chat.send_message(user_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Prompt enhancement failed: {e}")
            return user_prompt

# PART 5: Utility to open an image from URL
def open_image(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# PART 6: Convert sketch image to descriptive text using BLIP
def sketch_to_text(image_url):
    image = open_image(image_url)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# PART 7: Main pipeline - sketch to enhanced prompt to generated image
def generate_image_from_sketch(image_url):
    """
    Convert sketch to text, enhance prompt, then generate image and upload to Cloudinary.
    Always returns the Cloudinary URL.
    """
    try:
        # Step 1: Convert sketch to text
        sketch_text = sketch_to_text(image_url)
        print(f"\n[Sketch Description]\n{sketch_text}\n")

        # Step 2: Enhance prompt using Gemini
        enhancer = GeminiChatBot()
        enhanced_prompt = enhancer.enhance_prompt(sketch_text)
        print(f"\n[Enhanced Prompt]\n{enhanced_prompt}\n")

        # Step 3: Generate final image using Gemini image model
        client = genai_image.Client(api_key=IMAGE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=enhanced_prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        # Step 4: Extract image from response, save, and upload to Cloudinary
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                os.makedirs("static", exist_ok=True)
                image_filename = f"generated_image_{uuid.uuid4()}.png"
                image_path = os.path.join("static", image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_data)

                # Upload to Cloudinary
                cloud_url = upload_image_local(image_path)
                print("Uploaded to Cloudinary:", cloud_url)
                return cloud_url

        raise ValueError("No image data found in the response.")

    except Exception as e:
        print(f"Image generation failed: {e}")
        raise RuntimeError("Image generation failed.")

# PART 8: Main entry point for running the script
if __name__ == "__main__":
    sketch_input = input("🖌️ Enter URL to your sketch image: ")
    result = generate_image_from_sketch(sketch_input)
    print("\n[Cloudinary URL]\n" + result)
