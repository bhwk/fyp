#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "Retrieve LLAMA3 model..."
ollama pull llama3
echo "🟢 Done!"
echo "Retrieve Mistral-Nemo model..."
ollama pull mistral-nemo
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
