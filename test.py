from google import genai

client = genai.Client(api_key="AIzaSyBppF0OZ3VPmJ0rhrFAqiYpPYID3YXaiwk")

models = client.models.list()

for m in models:
    print(m.name)