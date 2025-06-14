from flask import Flask, request, render_template_string, jsonify, flash, redirect, url_for
import os
import tempfile
import subprocess
import numpy as np
from urllib.parse import urlparse
import requests
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Disable numba JIT compilation to avoid memory issues
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Import librosa after setting environment variable
import librosa

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class AccentAnalyzer:
    def __init__(self):
        # Initialize with a simple rule-based classifier
        # In production, this would be replaced with a trained ML model
        self.accent_patterns = {
            'american': {
                'formant_ratios': [1.2, 1.8, 2.4],
                'pitch_variance': 0.15,
                'speaking_rate': 150
            },
            'british': {
                'formant_ratios': [1.1, 1.6, 2.2],
                'pitch_variance': 0.12,
                'speaking_rate': 140
            },
            'australian': {
                'formant_ratios': [1.3, 1.9, 2.5],
                'pitch_variance': 0.18,
                'speaking_rate': 160
            },
            'indian': {
                'formant_ratios': [1.0, 1.4, 2.0],
                'pitch_variance': 0.20,
                'speaking_rate': 130
            },
            'canadian': {
                'formant_ratios': [1.15, 1.7, 2.3],
                'pitch_variance': 0.14,
                'speaking_rate': 145
            }
        }
    
    def extract_audio_features(self, audio_path):
        """Extract acoustic features from audio file - optimized for low memory"""
        try:
            # Load audio file with lower sample rate to save memory
            y, sr = librosa.load(audio_path, sr=16000, duration=30)  # Limit to 30 seconds
            
            # Extract features
            features = {}
            
            # Simplified pitch extraction to avoid numba issues
            try:
                # Use alternative pitch detection method
                hop_length = 512
                frame_length = 2048
                
                # Get spectral centroid (simpler than pitch tracking)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
                features['spectral_centroid'] = float(np.mean(spectral_centroids))
                features['spectral_variance'] = float(np.var(spectral_centroids))
                
                # Use spectral variance as pitch variance proxy
                features['pitch_variance'] = features['spectral_variance'] / features['spectral_centroid'] if features['spectral_centroid'] > 0 else 0.15
                features['mean_pitch'] = features['spectral_centroid']
                
            except Exception as pitch_error:
                print(f"Pitch extraction failed, using defaults: {pitch_error}")
                features['mean_pitch'] = 200.0
                features['pitch_variance'] = 0.15
                features['spectral_centroid'] = 200.0
            
            # MFCC features - reduce number to save memory
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=hop_length)  # Reduced from 13 to 5
                for i in range(5):
                    features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            except Exception as mfcc_error:
                print(f"MFCC extraction failed, using defaults: {mfcc_error}")
                for i in range(5):
                    features[f'mfcc_{i}'] = 0.0
            
            # Speaking rate estimation using zero crossing rate (simpler than onset detection)
            try:
                zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
                features['speaking_rate'] = float(np.mean(zcr) * sr / hop_length * 60)  # Rough approximation
                
                # Normalize speaking rate to reasonable range
                if features['speaking_rate'] > 300:
                    features['speaking_rate'] = 150
                elif features['speaking_rate'] < 50:
                    features['speaking_rate'] = 130
                    
            except Exception as rate_error:
                print(f"Speaking rate estimation failed: {rate_error}")
                features['speaking_rate'] = 140.0
            
            # Simplified formant approximation
            try:
                # Use spectral features as formant proxies
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
                
                rolloff_mean = float(np.mean(spectral_rolloff))
                bandwidth_mean = float(np.mean(spectral_bandwidth))
                centroid_mean = features['spectral_centroid']
                
                # Create formant ratios from spectral features
                if centroid_mean > 0:
                    features['formant_ratios'] = [
                        bandwidth_mean / centroid_mean if centroid_mean > 0 else 1.2,
                        rolloff_mean / centroid_mean if centroid_mean > 0 else 2.0,
                        rolloff_mean / bandwidth_mean if bandwidth_mean > 0 else 1.5
                    ]
                else:
                    features['formant_ratios'] = [1.2, 2.0, 1.6]
                    
            except Exception as formant_error:
                print(f"Formant estimation failed: {formant_error}")
                features['formant_ratios'] = [1.2, 2.0, 1.6]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features if everything fails
            return {
                'mean_pitch': 200.0,
                'pitch_variance': 0.15,
                'spectral_centroid': 200.0,
                'speaking_rate': 140.0,
                'formant_ratios': [1.2, 2.0, 1.6],
                'mfcc_0': 0.0, 'mfcc_1': 0.0, 'mfcc_2': 0.0, 'mfcc_3': 0.0, 'mfcc_4': 0.0
            }
    
    def classify_accent(self, features):
        """Classify accent based on extracted features"""
        if not features:
            return None, 0, "Could not extract audio features"
        
        scores = {}
        
        for accent, patterns in self.accent_patterns.items():
            score = 0
            
            # Compare formant ratios
            if 'formant_ratios' in features:
                formant_diff = sum(abs(a - b) for a, b in zip(features['formant_ratios'], patterns['formant_ratios']))
                formant_score = max(0, 1 - formant_diff / 3)  # Normalize
                score += formant_score * 0.4
            
            # Compare pitch variance
            if 'pitch_variance' in features:
                pitch_diff = abs(features['pitch_variance'] - patterns['pitch_variance'])
                pitch_score = max(0, 1 - pitch_diff / 0.5)  # Normalize
                score += pitch_score * 0.3
            
            # Compare speaking rate
            if 'speaking_rate' in features:
                rate_diff = abs(features['speaking_rate'] - patterns['speaking_rate'])
                rate_score = max(0, 1 - rate_diff / 100)  # Normalize
                score += rate_score * 0.3
            
            scores[accent] = score
        
        # Find best match
        best_accent = max(scores, key=scores.get)
        confidence = scores[best_accent] * 100
        
        # Generate explanation
        explanation = f"Analysis based on spectral patterns and speaking characteristics. "
        explanation += f"Detected speaking rate: {features.get('speaking_rate', 0):.0f} units/min, "
        explanation += f"Spectral variance: {features.get('pitch_variance', 0):.3f}"
        
        return best_accent, confidence, explanation

def download_video(url):
    """Download video from URL"""
    try:
        # Handle Google Drive links
        if 'drive.google.com' in url:
            # Extract file ID from Google Drive URL
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                # Convert to direct download URL
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Handle Google Drive download with confirmation
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            
            # Check if we need to handle download confirmation
            if 'download_warning' in response.text or 'virus scan warning' in response.text:
                # Look for the actual download link
                import re
                confirm_token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        confirm_token = value
                        break
                
                if confirm_token:
                    url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(url, stream=True, timeout=30)
        else:
            # Regular download
            response = requests.get(url, stream=True, timeout=30)
        
        response.raise_for_status()
        
        # Check if we actually got a video file
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            print("Error: Got HTML instead of video file. Check if the URL is a direct download link.")
            return None
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def extract_audio_from_video(video_path):
    """Extract audio from video using ffmpeg - optimized for low memory"""
    try:
        # Create temporary audio file
        audio_path = tempfile.mktemp(suffix='.wav')
        
        # Use ffmpeg to extract audio with optimization for memory
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Lower sample rate to save memory
            '-ac', '1',  # Mono
            '-t', '30',  # Limit to first 30 seconds
            '-y',  # Overwrite output
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return audio_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def is_valid_video_url(url):
    """Check if URL is a valid video URL"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Check for common video hosting domains or file extensions
        video_domains = ['loom.com', 'youtube.com', 'vimeo.com', 'dropbox.com', 'drive.google.com', 'googleapis.com']
        video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']
        
        domain_match = any(domain in parsed.netloc.lower() for domain in video_domains)
        extension_match = any(url.lower().endswith(ext) for ext in video_extensions)
        
        return domain_match or extension_match
    except:
        return False

# Initialize analyzer
analyzer = AccentAnalyzer()

# HTML template (same as before)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Accent Analysis Tool</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .result { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .error { background: #f5e8e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
        input[type="url"] { width: 100%; padding: 10px; margin: 10px 0; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #005a87; }
        .loading { display: none; color: #007cba; font-weight: bold; }
        .confidence-bar { background: #ddd; height: 20px; border-radius: 10px; margin: 10px 0; }
        .confidence-fill { background: #4CAF50; height: 100%; border-radius: 10px; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <h1>🎙️ Accent Analysis Tool</h1>
    <p>Analyze spoken English accents from video URLs</p>
    
    <div class="container">
        <form method="POST" onsubmit="showLoading()">
            <label for="video_url">Video URL:</label>
            <input type="url" id="video_url" name="video_url" placeholder="https://example.com/video.mp4 or Loom link" required>
            <br>
            <button type="submit">Analyze Accent</button>
        </form>
        <div class="loading" id="loading">🔄 Processing video... This may take a few moments.</div>
    </div>
    
    {% if result %}
    <div class="result">
        <h3>Analysis Results:</h3>
        <p><strong>Detected Accent:</strong> {{ result.accent.title() }}</p>
        <p><strong>Confidence Score:</strong> {{ "%.1f"|format(result.confidence) }}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {{ result.confidence }}%"></div>
        </div>
        <p><strong>Explanation:</strong> {{ result.explanation }}</p>
    </div>
    {% endif %}
    
    {% if error %}
    <div class="error">
        <strong>Error:</strong> {{ error }}
    </div>
    {% endif %}
    
    <div class="container">
        <h3>Supported URLs:</h3>
        <ul>
            <li>Direct video links (.mp4, .avi, .mov, etc.)</li>
            <li>Loom recordings</li>
            <li>Google Drive links</li>
            <li>Other public video hosting platforms</li>
        </ul>
        
        <h3>How it works:</h3>
        <p>The tool downloads the video, extracts audio, analyzes acoustic features like spectral patterns, 
        and speaking characteristics, then classifies the accent using pattern matching.</p>
        
        <p><strong>Note:</strong> Processing is optimized for cloud deployment and analyzes the first 30 seconds of audio.</p>
    </div>
    
    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/test-ffmpeg')
def test_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return f"✅ FFmpeg is working!<br><pre>{result.stdout[:200]}</pre>"
        else:
            return f"❌ FFmpeg error: {result.stderr}"
    except FileNotFoundError:
        return "❌ FFmpeg not found"
    except Exception as e:
        return f"❌ Error: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        
        if not video_url:
            return render_template_string(HTML_TEMPLATE, error="Please provide a video URL")
        
        if not is_valid_video_url(video_url):
            return render_template_string(HTML_TEMPLATE, error="Please provide a valid video URL")
        
        try:
            # Download video
            print(f"Downloading video from: {video_url}")
            video_path = download_video(video_url)
            if not video_path:
                return render_template_string(HTML_TEMPLATE, error="Failed to download video. Please check the URL.")
            
            # Extract audio
            print("Extracting audio...")
            audio_path = extract_audio_from_video(video_path)
            if not audio_path:
                os.unlink(video_path)
                return render_template_string(HTML_TEMPLATE, error="Failed to extract audio from video. Make sure ffmpeg is installed.")
            
            # Analyze accent
            print("Analyzing accent...")
            features = analyzer.extract_audio_features(audio_path)
            accent, confidence, explanation = analyzer.classify_accent(features)
            
            # Clean up temporary files
            os.unlink(video_path)
            os.unlink(audio_path)
            
            if accent:
                result = {
                    'accent': accent,
                    'confidence': confidence,
                    'explanation': explanation
                }
                return render_template_string(HTML_TEMPLATE, result=result)
            else:
                return render_template_string(HTML_TEMPLATE, error="Failed to analyze accent from the audio")
                
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"An error occurred: {str(e)}")
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.get_json()
    
    if not data or 'video_url' not in data:
        return jsonify({'error': 'video_url is required'}), 400
    
    video_url = data['video_url']
    
    if not is_valid_video_url(video_url):
        return jsonify({'error': 'Invalid video URL'}), 400
    
    try:
        # Process video
        video_path = download_video(video_url)
        if not video_path:
            return jsonify({'error': 'Failed to download video'}), 400
        
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            os.unlink(video_path)
            return jsonify({'error': 'Failed to extract audio'}), 400
        
        # Analyze
        features = analyzer.extract_audio_features(audio_path)
        accent, confidence, explanation = analyzer.classify_accent(features)
        
        # Cleanup
        os.unlink(video_path)
        os.unlink(audio_path)
        
        if accent:
            return jsonify({
                'accent': accent,
                'confidence': confidence,
                'explanation': explanation
            })
        else:
            return jsonify({'error': 'Failed to analyze accent'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)