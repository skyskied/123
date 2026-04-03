import urllib.request

url = "https://huggingface.co/junjiang/GestureFace/resolve/main/blaze_face_short_range.tflite"

urllib.request.urlretrieve(url, "blaze_face_short_range.tflite")

print("Model downloaded successfully.")
