import os
import pickle
import cv2
from utils import extract_face, get_embedding

DATASET_PATH = "data/raw_faces"
EMBEDDINGS_PATH = "embeddings/embeddings.pkl"

embeddings = {}
total_faces = 0

print("Scanning dataset...")

for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_dir):
        continue

    print(f"\nProcessing person: {person_name}")
    person_embeddings = []

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        print(f"  Reading image: {image_name}")

        image = cv2.imread(image_path)

        if image is None:
            print("   ❌ Image could not be read")
            continue

        face = extract_face(image)

        if face is None:
            print("   ❌ No face detected")
            continue

        embedding = get_embedding(face)
        person_embeddings.append(embedding)
        total_faces += 1
        print("   ✅ Face detected & embedding created")

    if len(person_embeddings) > 0:
        embeddings[person_name] = person_embeddings

if len(embeddings) == 0:
    print("\n❌ No embeddings created. Check images.")
else:
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print("\n✅ Embedding generation completed")
    print("Total persons:", len(embeddings))
    print("Total face samples:", total_faces)
