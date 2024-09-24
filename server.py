import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

VALID_LOCATIONS = [
    'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California', 
    'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California', 
    'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California', 
    'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California', 
    'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah', 
    'San Diego, California', 'Tucson, Arizona'
]
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Extract query parameters
            query = parse_qs(environ.get("QUERY_STRING", ""))
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]

            # Filter and sort reviews
            filtered_reviews = self.filter_reviews(location, start_date, end_date)
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
            
            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
            
            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]
        if environ["REQUEST_METHOD"] == "POST":
            # Get content length and read the request body
            content_length = environ.get('CONTENT_LENGTH')
            content_length = int(content_length) if content_length else 0
            input_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            params = parse_qs(input_data)

            # Extract parameters
            location = params.get('Location', [None])[0]
            review_body = params.get('ReviewBody', [None])[0]

            # Validate parameters
            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Location and ReviewBody are required"}).encode("utf-8")]

            if location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Invalid Location"}).encode("utf-8")]

            # Create new review entry without sentiment analysis
            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp,
            }

            reviews.append(new_review)

            # Create response body and set headers
            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        # Handle unknown methods
        start_response("404 Not Found", [("Content-Type", "application/json")])
        return [json.dumps({"error": "Not Found"}).encode("utf-8")]

    def filter_reviews(self, location, start_date, end_date):
        filtered = []
        for review in reviews:
            if location and review['Location'] != location:
                continue
            if start_date and datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') < datetime.strptime(start_date, '%Y-%m-%d'):
                continue
            if end_date and datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') > datetime.strptime(end_date, '%Y-%m-%d'):
                continue
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            filtered.append(review)
        return filtered

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()