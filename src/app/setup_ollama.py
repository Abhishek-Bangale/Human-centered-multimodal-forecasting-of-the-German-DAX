import subprocess
import os
import time
import requests

def setup_ollama():
    print("🤖 Setting up Ollama for AI explanations...")
    
    # Path to Ollama executable
    ollama_path = os.path.expanduser("~/AppData/Local/Programs/Ollama/ollama.exe")
    
    if not os.path.exists(ollama_path):
        print(f"❌ Ollama not found at {ollama_path}")
        print("Please install Ollama from https://ollama.ai")
        return False
    
    try:
        # Start Ollama service
        print("🚀 Starting Ollama service...")
        subprocess.Popen([ollama_path, "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        time.sleep(3)
        
        # Check if service is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service is running")
            else:
                print("⚠️ Ollama service may not be fully started")
        except:
            print("⚠️ Ollama service may not be fully started")
        
        # Download phi4-mini model
        print("📥 Downloading phi4-mini model...")
        result = subprocess.run([ollama_path, "pull", "phi4-mini:latest"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ phi4-mini model downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download model: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up Ollama: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_ollama()
    if success:
        print("\n🎉 Ollama setup completed! AI explanations should now work.")
    else:
        print("\n💥 Ollama setup failed. Explanations will be disabled.") 