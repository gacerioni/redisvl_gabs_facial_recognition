import base64
import os
import numpy as np
from PIL import Image
from redisvl.index import SearchIndex
from redis import Redis
import requests
from io import BytesIO
from deepface import DeepFace
from urllib.parse import urlparse


# Global Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_connection(redis_url):
    """
    Create a Redis connection from a URL.
    """
    parsed_url = urlparse(redis_url)
    return Redis(
        host=parsed_url.hostname,
        port=parsed_url.port or 6379,
        password=parsed_url.password,
        decode_responses=False  # Binary storage enabled
    )


def create_redis_index(client):
    """Define and create the Redis index."""
    schema = {
        "index": {
            "name": "face_recognition",
            "prefix": "face_docs",
        },
        "fields": [
            {"name": "name", "type": "tag"},
            {"name": "photo_reference", "type": "text"},
            {"name": "photo_binary", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 128,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                }
            }
        ]
    }
    index = SearchIndex.from_dict(schema)
    index.set_client(client)
    index.create(overwrite=True)
    return index


def load_remote_image(url):
    """Download and preprocess an image from a remote URL."""
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img


def generate_embedding(image_path):
    """Generate a real embedding using DeepFace."""
    embedding = DeepFace.represent(image_path, model_name="Facenet")
    return np.array(embedding[0]["embedding"], dtype=np.float32)


def inject_data_into_redis(local_db, index):
    """Inject images, embeddings, and metadata into Redis."""
    for name, url in local_db.items():
        # Load the image
        img = load_remote_image(url)

        # Convert image to binary
        img_binary = BytesIO()
        img.save(img_binary, format="JPEG")  # Ensure consistent format
        img_binary.seek(0)
        raw_binary_data = img_binary.read()  # Raw bytes

        # Encode the binary data to Base64 for storage
        encoded_binary_data = base64.b64encode(raw_binary_data).decode("utf-8")

        # Debug: Save the binary locally for comparison
        with open(f"{name}_original_debug.jpg", "wb") as f:
            f.write(raw_binary_data)

        # Generate embedding
        embedding = generate_embedding(url)

        # Store the embedding, photo reference, and Base64-encoded binary data in Redis
        index.load([{
            "name": name,
            "photo_reference": url,
            "photo_binary": encoded_binary_data,  # Store as Base64-encoded text
            "embedding": embedding.tobytes()
        }])

        print(f"Stored {name} in Redis with encoded binary size: {len(encoded_binary_data)}")


def query_redis(target_image_path, index, client, display_images=True):
    """Query Redis with a target image and display results."""
    # Generate embedding for the target image
    target_embedding = generate_embedding(target_image_path)

    # Create a vector query
    from redisvl.query import VectorQuery
    query = VectorQuery(
        vector=target_embedding.tolist(),
        vector_field_name="embedding",
        return_fields=["name", "photo_reference", "vector_distance", "photo_binary"],
        num_results=5
    )

    # Execute query
    results = index.query(query)

    # Parse and sort results by distance
    matches = []
    for result in results:
        name = result["name"]
        photo_url = result["photo_reference"]
        # Decode Base64 back to raw binary
        photo_binary = base64.b64decode(result["photo_binary"])
        distance = float(result["vector_distance"])

        matches.append({
            "name": name,
            "photo_url": photo_url,
            "photo_binary": photo_binary,
            "vector_distance": distance
        })

    # Sort matches by vector_distance
    matches.sort(key=lambda x: x["vector_distance"])

    # Display rankings
    print("\n--- Ranking ---")
    for idx, match in enumerate(matches, start=1):
        print(f"{idx}. Name: {match['name']}, Distance: {match['vector_distance']:.2f}")
        print(f"   URL: {match['photo_url']}")

    # Display best match
    if matches:
        best_match = matches[0]
        print("\n--- Best Match ---")
        print(f"Name: {best_match['name']}")
        print(f"Image URL: {best_match['photo_url']}")
        print(f"Match Distance: {best_match['vector_distance']:.2f}")

        if display_images:
            try:
                # Save the binary retrieved from Redis for debugging
                retrieved_binary = best_match["photo_binary"]
                print(f"Retrieved binary size: {len(retrieved_binary)}")
                with open(f"{best_match['name']}_retrieved_debug.jpg", "wb") as f:
                    f.write(retrieved_binary)

                # Decode and display image
                img = Image.open(BytesIO(retrieved_binary))
                img.show()  # Display image
                print(f"Decoded image for {best_match['name']} displayed.")
            except Exception as e:
                print(f"Error decoding image for {best_match['name']}: {e}")
    else:
        print("\nNo match found.")

def clear_face_docs(client):
    """Delete all face_docs:* hashes in Redis."""
    keys = client.keys("face_docs:*")
    if keys:
        client.delete(*keys)
        print(f"Deleted {len(keys)} keys matching 'face_docs:*'.")
    else:
        print("No keys matching 'face_docs:*' found.")


def main():
    """Main function to orchestrate Redis face recognition steps."""
    # Connect to Redis
    client = get_redis_connection(REDIS_URL)

    # Clear existing face_docs
    clear_face_docs(client)

    # Create the Redis index
    index = create_redis_index(client)

    # Remote image URLs
    local_db = {
        'angelina': 'https://github.com/serengil/deepface/raw/master/tests/dataset/img2.jpg',
        'jennifer': 'https://github.com/serengil/deepface/raw/master/tests/dataset/img56.jpg',
        'scarlett': 'https://github.com/serengil/deepface/raw/master/tests/dataset/img49.jpg',
        'katy': 'https://github.com/serengil/deepface/raw/master/tests/dataset/img42.jpg',
        'marissa': 'https://github.com/serengil/deepface/raw/master/tests/dataset/img23.jpg'
    }

    # Inject data into Redis
    inject_data_into_redis(local_db, index)
    print("Data successfully injected into Redis.")

    # Test querying with Angelina Jolie's same picture
    print("\n--- Testing with Angelina Jolie's Same Picture ---")
    query_redis("https://github.com/serengil/deepface/raw/master/tests/dataset/img2.jpg", index, client)

    # Now we try to find a real angelina jolie by using a different image from the web
    print("\n--- Testing with Angelina Jolie's Different Picture ---")
    query_redis("https://i.pinimg.com/474x/e4/95/1b/e4951b2a165fd62b5a468fed250fb941.jpg", index, client)


if __name__ == "__main__":
    main()