import torch
import redis
import hashlib
import soundfile as sf
import os
import logging
import tempfile
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from sparktts.utils.token_parser import LEVELS_MAP_UI
from cli.SparkTTS import SparkTTS
from segmentation_and_memory import MemoryManager  # Import your MemoryManager class
import faiss
import numpy as np
from datetime import datetime
from segmentation_and_memory import segmentation
import nltk
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Static voice parameters
FIXED_GENDER = "female"
FIXED_AGE = "Youth-Adult"
FIXED_PITCH = "3"
FIXED_SPEED = "3"
FIXED_EMOTION = "NEUTRAL"
TEMP_DIR = "temp_audio"

# Initialize Spark-TTS model
def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device="cpu"):
    device = torch.device("cpu" if device == "cpu" else f"cuda:{device}")
    return SparkTTS(model_dir, device)

# def real_time_tts(model, text_input, memory_manager):
#     # First, try to find a similar session in the memory manager (FAISS + Redis)
#     segments = text_input.split(",")
#     all_matched_texts = []  
#     matched_text, audio_file = memory_manager.search_session(text_input)
    
#     if matched_text and audio_file and os.path.exists(audio_file):
#         logging.info(f"Returning cached audio: {audio_file}")
#         return audio_file  # Return the audio file if found in cache
    
#     # If no match is found, generate new audio
#     pitch_val = LEVELS_MAP_UI[int(FIXED_PITCH)]
#     speed_val = LEVELS_MAP_UI[int(FIXED_SPEED)]
    
#     try:
#         with torch.no_grad():
#             wav = model.inference(
#                 text=text_input,
#                 gender=FIXED_GENDER,
#                 pitch=pitch_val,
#                 speed=speed_val
#             )
#             if isinstance(wav, torch.Tensor):  # Ensure it's a numpy array
#                 wav = wav.cpu().numpy()
#     except Exception as e:
#         logging.error(f"TTS inference failed: {str(e)}")
#         return None
    
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     temp_path = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
#     sf.write(temp_path, wav, 16000)
    
#     # Store the generated session in memory (FAISS and Redis)
#     memory_manager.store_session(text_input, temp_path)
#     logging.info(f"Generated audio: {temp_path}")
    
#     return temp_path

def real_time_tts(model, text_input, memory_manager):
    # Split the text into segments by sentences or logical chunks
    segments = segmentation.segment_text(text_input, delimiter=',') 
    all_matched_texts = []  # List to keep track of all the dynamically processed parts
    all_audio_paths = []  # List to store paths of all generated audio files

 
    for segment in segments:
        # Strip extra spaces from segment
        segment = segment.strip()
        
        if not segment:
            continue
        
        # First, try to find a similar session in the memory manager (FAISS + Redis)
        matched_text, audio_file = memory_manager.search_session(segment)
        
        if matched_text and audio_file and os.path.exists(audio_file):
            # If part is found in cache, print the cached part
            all_matched_texts.append(f"Cached part: '{matched_text}'")
            logging.info(f"Returning cached audio: {audio_file}")
            all_audio_paths.append(audio_file)  # Add the cached audio file to the list
        else:
            # If no match is found, generate new audio
            pitch_val = LEVELS_MAP_UI[int(FIXED_PITCH)]
            speed_val = LEVELS_MAP_UI[int(FIXED_SPEED)]
            
            try:
                with torch.no_grad():
                    wav = model.inference(
                        text=segment,
                        gender=FIXED_GENDER,
                        pitch=pitch_val,
                        speed=speed_val
                    )
                    if isinstance(wav, torch.Tensor):  # Ensure it's a numpy array
                        wav = wav.cpu().numpy()
            except Exception as e:
                logging.error(f"TTS inference failed for segment '{segment}': {str(e)}")
                return None

            
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_path = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
            sf.write(temp_path, wav, 16000)
            
            # Store the generated session in memory (FAISS and Redis)
            memory_manager.store_session(segment, temp_path)
            all_matched_texts.append(f"Generated part: '{segment}'")
            logging.info(f"Generated audio: {temp_path}")
            all_audio_paths.append(temp_path)  # Add the generated audio file to the list
    
    # Print all dynamically matched/generated parts
    logging.info("Processed text parts:")
    for part in all_matched_texts:
        logging.info(part)

    # If we have multiple audio parts, concatenate them into a single audio file
    if len(all_audio_paths) > 1:
        combined_audio = []
        for path in all_audio_paths:
            data, samplerate = sf.read(path)
            combined_audio.append(data)

        # Concatenate all parts
        combined_audio = np.concatenate(combined_audio, axis=0)

        # Save the combined audio to a new file
        combined_path = os.path.join(TEMP_DIR, f"combined_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
        sf.write(combined_path, combined_audio, samplerate)
        logging.info(f"Combined audio saved to: {combined_path}")
        return combined_path  # Return the combined audio file
    else:
        # If only one audio part exists, return that directly
        return all_audio_paths[0]

# Gradio UI Integration
def build_gradio_ui(memory_manager, model_dir="pretrained_models/Spark-TTS-0.5B", device="cpu"):
    model = initialize_model(model_dir, device)
    cached_text = ""
    
    def gradio_callback(text):
        nonlocal cached_text
        new_text = text[len(cached_text):].strip() if cached_text and text.startswith(cached_text) else text.strip()
        if not new_text:
            logging.info("No new text")
            return None
        audio_file = real_time_tts(model, new_text, memory_manager)
        cached_text = text.strip()
        return audio_file
    
    with gr.Blocks(title="Real-Time TTS with Memory") as demo:
        gr.Markdown("# Real-Time TTS with Caching")
        text_input = gr.Textbox(label="Input Text", lines=3, placeholder="Type here...")
        audio_output = gr.Audio(label="Output", type="filepath", autoplay=True)
        gr.Button("Generate").click(gradio_callback, inputs=text_input, outputs=audio_output)
    
    return demo


if __name__ == "__main__":
    memory_manager = MemoryManager()
    demo = build_gradio_ui(memory_manager)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)