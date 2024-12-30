import os.path

import ollama


class LlamaOCR:
    def __init__(self, model='llama3.2-vision'):
        self.model = model

    def extract_text(self, image_path):
        """
        Extract text from an image file using the Llama OCR model.

        :param image_path: Path to the image file.
        :return: Extracted text in Markdown format.
        """
        try:
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()

            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': """Analyze the text in the provided image. Extract all readable content
                                and present it in a structured Markdown format that is clear, concise, 
                                and well-organized. Ensure proper formatting (e.g., headings, lists, or
                                code blocks) as necessary to represent the content effectively.""",
                    'images': [image_bytes]
                }]
            )
            return response.message.content
        except Exception as e:
            return f"Error processing image: {str(e)}"


# Usage example
if __name__ == "__main__":
    ocr = LlamaOCR()
    image_path = os.path.join(os.curdir, "images", "IMG_20240929_0004.jpg")  # Replace with your image file path
    result = ocr.extract_text(image_path)
    print(result)
