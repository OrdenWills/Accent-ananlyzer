# 🎙️ Accent Analysis Tool

A Flask web application that analyzes English accents from video URLs using acoustic feature extraction and machine learning techniques.

## Features

- **Accent Classification**: Detects American, British, Australian, Indian, and Canadian English accents
- **Video Processing**: Downloads videos from URLs and extracts audio for analysis
- **Confidence Scoring**: Provides confidence percentages for accent predictions
- **Web Interface**: User-friendly web interface for easy testing
- **API Endpoint**: RESTful API for programmatic access
- **Multiple URL Support**: Works with Google Drive, Dropbox, Loom, and direct video links

## How It Works

1. **Audio Extraction**: Downloads video from URL and extracts audio using FFmpeg
2. **Feature Analysis**: Extracts acoustic features including:
   - Pitch patterns and variance
   - Formant frequency ratios
   - Speaking rate estimation
   - Spectral characteristics
   - MFCC coefficients
3. **Accent Classification**: Compares extracted features against known accent patterns
4. **Results**: Returns detected accent with confidence score and explanation

## Quick Start

### Option 1: Try the Live Demo
Visit the deployed application: [Your Render URL will go here]

### Option 2: Run Locally

#### Prerequisites
- Python 3.8+
- FFmpeg installed on your system

#### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/accent-analyzer
cd accent-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit `http://localhost:5000` in your browser.

## Testing the Application

### Sample URLs for Testing

**Google Drive (recommended):**
```
https://drive.google.com/uc?export=download&id=17keuLbbvPsTRJYUeIVQOiyQM3D8wwDrm
```

**Sample Video:**
```
https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4
```

**For Your Own Videos:**

1. **Google Drive**: 
   - Upload video to Google Drive
   - Share → Anyone with link can view
   - Copy link, extract file ID from: `https://drive.google.com/file/d/FILE_ID/view`
   - Use: `https://drive.google.com/uc?export=download&id=FILE_ID`

2. **Dropbox**:
   - Upload video, get share link
   - Change `?dl=0` to `?dl=1` at the end

3. **Direct Links**: Any publicly accessible video URL (.mp4, .avi, .mov, etc.)

### Web Interface Usage

1. Open the application in your browser
2. Paste a video URL in the input field
3. Click "Analyze Accent"
4. Wait for processing (may take 30-60 seconds)
5. View results with accent classification and confidence score

### API Usage

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "your-video-url-here"}'
```

Response:
```json
{
  "accent": "american",
  "confidence": 78.5,
  "explanation": "Analysis based on pitch patterns, formant frequencies, and speaking rate. Detected speaking rate: 145 words/min, Pitch variance: 0.168"
}
```

## Supported Accents

- **American English**: Standard North American accent
- **British English**: Received Pronunciation and general British accents
- **Australian English**: General Australian accent
- **Indian English**: Indian-influenced English pronunciation
- **Canadian English**: Canadian accent patterns

## Technical Details

### Architecture
- **Backend**: Flask (Python)
- **Audio Processing**: Librosa, FFmpeg
- **Feature Extraction**: Pitch analysis, formant estimation, spectral features
- **Classification**: Pattern matching with acoustic feature comparison

### Key Dependencies
- Flask: Web framework
- Librosa: Audio analysis
- NumPy: Numerical computations
- Requests: HTTP handling
- FFmpeg: Video/audio processing

### Deployment
The application is designed to run on cloud platforms like Render, Heroku, or similar PaaS providers.

## Development

### Project Structure
```
accent-analyzer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── runtime.txt           # Python version for deployment
└── Procfile              # Process file for deployment
```

### Local Development
```bash
# Install in development mode
pip install -r requirements.txt

# Run with debug mode
python app.py
```

### Adding New Accents
To add support for new accents, update the `accent_patterns` dictionary in the `AccentAnalyzer` class with the new accent's characteristic features.

## Limitations

- **English Only**: Currently supports English accents only
- **Audio Quality**: Results depend on audio quality and clarity
- **Simple Classification**: Uses pattern matching rather than deep learning
- **Processing Time**: Video processing may take 30-60 seconds depending on file size

## Troubleshooting

### Common Issues

**"Failed to download video"**
- Ensure the URL is publicly accessible
- Check if the URL is a direct download link
- Try the suggested Google Drive or Dropbox formats

**"moov atom not found"**
- The downloaded file is not a valid video (possibly HTML)
- Use the direct download URL format for Google Drive

**"Failed to extract audio"**
- Ensure FFmpeg is installed on the system
- Check if the video file is corrupted

### Getting Help
- Check the console output for detailed error messages
- Ensure all dependencies are properly installed
- Verify FFmpeg is available in your system PATH

## Performance Notes

- Processing time varies with video length (typically 30-60 seconds)
- Longer videos may provide more accurate results
- Clear speech with minimal background noise gives best results

## Future Enhancements

- Deep learning model for improved accuracy
- Support for more accent varieties
- Real-time audio processing
- Batch processing capabilities
- Enhanced confidence scoring algorithms

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Note**: This tool is designed for educational and research purposes. Accent classification results should be considered estimates and may not be 100% accurate.