import os

# Path to your dataset folder (change this!)
dataset_path = r"D:\ML\Plant_disease (not to be presented)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

# Get all folder names (classes)
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

# Save to file
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"✅ Saved {len(class_names)} class names to class_names.txt")
