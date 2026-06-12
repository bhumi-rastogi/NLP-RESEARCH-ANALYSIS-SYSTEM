#!/bin/bash
# Setup script for ResearchScope

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

echo ""
echo "Setup complete! Run the app with:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
