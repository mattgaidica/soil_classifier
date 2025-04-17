# USCS Soil Classification Tool

A web-based tool for classifying soils according to the Unified Soil Classification System (USCS).

## Features

- Input soil grain size distribution data
- Input Atterberg limits (LL, PL)
- Automatic soil classification according to ASTM D2487
- Interactive grain size distribution plot
- Detailed classification results with key parameters
- Sample data for testing and demonstration

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd uscs-soil-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Development

Run the Streamlit app locally:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment

This app is ready for deployment on Streamlit Cloud:

1. Push your code to GitHub:
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Deploy your app by selecting your repository
5. Select `app.py` as the main file
6. Click "Deploy!"

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies

## Usage

1. Enter soil grain size distribution data (percent passing)
2. Input Atterberg limits if available
3. Click "Classify Soil" to get results
4. Use "Load Sample Data" to test the tool

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this code for any purpose.

## Acknowledgments

- Based on ASTM D2487 Standard
- Created for USAF CE applications 