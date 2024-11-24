import kagglehub

# Download latest version
path = kagglehub.dataset_download("yveslr/open-famous-people-faces")

print("Path to dataset files:", path)