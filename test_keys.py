import google.generativeai as genai
import sys

API_KEYS = [
    "AIzaSyDmZ22YZofRf3NeYsJSIswrDXXCyTcmcRU",
    "AIzaSyCghyHFi8DWSy026L1jB_OEmd-QWcSezCY",
    "AIzaSyALNXMUxpVnDQ9-jVlVo02rXjLC0hwCSy0"
]

for key in API_KEYS:
    print(f"\nTesting key: {key}")
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content("say hi")
        print("SUCCESS! Output length:", len(res.text))
    except Exception as e:
        print(f"FAILED: {str(e)}")
