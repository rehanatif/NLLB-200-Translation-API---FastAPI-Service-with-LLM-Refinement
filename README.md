# 🌍 NLLB-200 Translation API

> A production-ready FastAPI translation service powered by Meta's NLLB-200 model with optional LLM refinement for natural, UI-friendly translations.

## ✨ Features

- **🌐 NLLB-200 Translation**: Fast, accurate translation using Meta's NLLB-200 model (200+ languages)
- **✨ Optional LLM Refinement**: Context-aware refinement for natural, culturally appropriate UI/UX translations
- **📦 Multiple Input Formats**: Supports single strings, arrays, and dictionaries
- **⚡ Batch Processing**: Efficient batch translation with optimized memory management
- **🐳 Docker Support**: Easy deployment with Docker
- **🚀 RESTful API**: Comprehensive error handling and health checks
- **🎯 GPU Acceleration**: CUDA support for faster inference
- **🔍 Language Discovery**: Built-in endpoint to discover available languages

## 🚀 Quick Start

### Option 1: Using Startup Scripts (Easiest)

**Windows:**
```bash
start_server.bat
```

**Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

### Option 2: Docker

```bash
docker build -t translation-api .
docker run -p 8005:8005 translation-api
```

### Option 3: Direct Python

```bash
python main.py
```

### Option 4: Using Uvicorn Directly

```bash
uvicorn main:app --host 0.0.0.0 --port 8005
```

## 📡 API Endpoints

### POST `/translate`
Translate text with optional refinement.

**Request Body:**
```json
{
  "text": "Save changes",
  "lang": "jpn_Jpan",
  "is_context_friendly": false
}
```

**Parameters:**
- `text` (required): String, array of strings, or dictionary
- `lang` (required): Target language code (e.g., `jpn_Jpan`, `spa_Latn`, `fra_Latn`)
- `is_context_friendly` (optional): Boolean, default `false`. If `true`, applies LLM refinement for UI/UX localization.

**Response:**
```json
{
  "success": true,
  "translation": "変更を保存"
}
```

### GET `/health`
Check service health and model status.

**Response:**
```json
{
  "status": "healthy",
  "translation_model": "facebook/nllb-200-distilled-600M",
  "refinement_model": "microsoft/phi-2",
  "refinement_available": true,
  "device": "cuda"
}
```

### GET `/languages`
Get list of available language codes.

## 💡 Use Cases

- **Application Localization**: Translate UI text for international applications
- **UI/UX Translation**: Natural, context-aware translations for buttons, labels, and messages
- **Batch Content Translation**: Efficiently translate multiple texts at once
- **Multi-language API Integration**: Integrate translation capabilities into your services
- **Dictionary Translation**: Preserve keys while translating values

## 🔧 Tech Stack

- **FastAPI** - Modern Python web framework
- **Transformers** - Hugging Face model integration
- **PyTorch** - Deep learning backend
- **Meta NLLB-200** - Translation model (200+ languages)
- **Microsoft Phi-2** - Refinement model (optional, for UI-friendly translations)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 📦 Installation

### 1. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**No additional libraries needed!** All required packages are already in `requirements.txt`.

## 🎯 First Run

On the first run, the service will automatically download:
- **NLLB-200 model** (~1.2GB) - Required for translation
- **Phi-2 model** (~2.7GB) - Optional, for refinement feature

**Total download: ~3.9GB**

This may take 10-30 minutes depending on your internet speed. The models are cached locally after the first download.

## 📖 Usage Examples

### Basic Translation (No Refinement)
```bash
curl -X POST "http://localhost:8005/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "lang": "jpn_Jpan"
  }'
```

### Translation with Refinement (UI-Friendly)
```bash
curl -X POST "http://localhost:8005/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Save changes",
    "lang": "jpn_Jpan",
    "is_context_friendly": true
  }'
```

### Batch Translation
```bash
curl -X POST "http://localhost:8005/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": ["Save", "Cancel", "Delete"],
    "lang": "jpn_Jpan",
    "is_context_friendly": true
  }'
```

### Dictionary Translation
```bash
curl -X POST "http://localhost:8005/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": {
      "save": "Save",
      "cancel": "Cancel",
      "delete": "Delete"
    },
    "lang": "jpn_Jpan",
    "is_context_friendly": true
  }'
```

## 🌐 Language Codes

Common language codes:
- `eng_Latn` - English
- `jpn_Jpan` - Japanese
- `spa_Latn` - Spanish
- `fra_Latn` - French
- `deu_Latn` - German
- `zho_Hans` - Simplified Chinese
- `zho_Hant` - Traditional Chinese
- `ara_Arab` - Arabic
- `por_Latn` - Portuguese
- `rus_Cyrl` - Russian
- `kor_Hang` - Korean

Use `/languages` endpoint to get the full list.

## ⚡ Performance

- **Translation Speed**: ~0.5-2 seconds per request (depending on text length)
- **With Refinement**: Adds ~1-3 seconds per request
- **Batch Processing**: Optimized for multiple texts
- **Timeout**: 55 seconds (configured to stay under 60s client timeout)

## 🔧 Troubleshooting

### Model Download Fails
- Check internet connection
- Ensure sufficient disk space (~5GB)
- Models are cached in `~/.cache/huggingface/`

### Out of Memory
- Reduce `MAX_BATCH_SIZE` in `main.py` (default: 30)
- Use CPU instead of GPU (slower but uses less memory)
- Process smaller batches

### Refinement Model Not Loading
- The service will continue without refinement
- Check logs for specific error messages
- Ensure you have enough RAM/VRAM (refinement model needs ~3GB)

## 📝 Notes

- The service runs on `0.0.0.0:8005` by default (accessible from network)
- For local-only access, change to `127.0.0.1:8005` in `main.py`
- GPU acceleration significantly improves speed
- First request after startup may be slower (model warm-up)

## 📄 License

This project uses models from Meta (NLLB-200) and Microsoft (Phi-2). Please refer to their respective licenses for usage terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Star History

If you find this project useful, please consider giving it a star!


## Laravel App Changes

The "Laravel Changes" folder includes all required files to integrate and ensure compatibility between your Laravel application and the FastAPI-based NLLB-200 AI translation model, enabling synchronization of content into a secondary language. You need to add that folder's files past into your laravel app.
