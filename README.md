# VisualGPT Toy Project

A visual AI orchestration project that combines image processing, OCR, and AI interactions.

## Features

- **Image Processing**: Uses OpenCV and Pillow for image manipulation
- **OCR Text Extraction**: Extracts text from images using Tesseract
- **AI Integration**: Connects with OpenAI's API for intelligent responses
- **Modular Design**: Separate orchestrator and tools for clean architecture

## Setup

### Prerequisites

- Python 3.9+
- Tesseract OCR engine
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shoreline/visualGPT_toy.git
   cd visualGPT_toy
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv vchat
   source vchat/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pillow opencv-python numpy "pydantic==2.*" rich openai pytesseract
   ```

4. **Install Tesseract OCR:**
   ```bash
   brew install tesseract
   ```

5. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

```bash
python orchestrator.py --goal "Your goal here" --image "path/to/image.jpg"
```

## Project Structure

```
visualGPT_toy/
├── orchestrator.py    # Main orchestration logic
├── tools.py          # Tool implementations
├── cat.jpg           # Sample image
├── artifacts/        # Generated outputs (ignored by git)
└── README.md         # This file
```

## Dependencies

- `pillow` - Image processing
- `opencv-python` - Computer vision
- `numpy` - Numerical operations
- `pydantic` - Data validation
- `rich` - Beautiful terminal output
- `openai` - AI/LLM interactions
- `pytesseract` - OCR text extraction

## License

This is a toy project for learning purposes.

## Contributing

Feel free to fork and experiment with the code!
