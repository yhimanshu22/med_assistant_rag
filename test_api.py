import requests
import time

URL = "http://127.0.0.1:8000/query"

def test_api():
    print(f"Testing API at {URL}...")
    
    payload = {
        "question": "What is the purpose of this RAG system?"
    }
    
    try:
        start = time.time()
        response = requests.post(URL, json=payload)
        end = time.time()
        
        if response.status_code == 200:
            print("\n✅ Success!")
            data = response.json()
            print(f"Time: {data.get('total_time')}")
            print(f"Question: {data.get('question')}")
            print(f"Answer: {data.get('answer')}")
            
            if data.get('source_documents'):
                print("\nSources:")
                for doc in data['source_documents']:
                    print(f"- {doc.get('metadata', {}).get('source', 'Unknown')}")
        else:
            print(f"\n❌ Global Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        print("Is the server running?")

if __name__ == "__main__":
    test_api()
