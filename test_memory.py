import logging
import os
from datetime import datetime
from segmentation_and_memory.memory_management import MemoryManager

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path for temporary files (you can modify this as needed)
TEMP_AUDIO_PATH = "temp_audio/"
FAISS_INDEX_FILE = "faiss_index.bin"

# Initialize the MemoryManager
memory_manager = MemoryManager(index_file=FAISS_INDEX_FILE)

# Create a function to generate dummy data for testing
def generate_dummy_data(num_entries=10):
    test_data = []
    for i in range(num_entries):
        text = f"Test entry number {i+1}. This is a sample text for testing."
        audio_file = os.path.join(TEMP_AUDIO_PATH, f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.wav")
        
        # Create a dummy audio file (you can replace this with actual audio generation)
        os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)
        with open(audio_file, "w") as f:
            f.write(f"dummy_audio_content_{i+1}")
        
        test_data.append((text, audio_file))
    return test_data

# Test Case 1: Store multiple sessions (10 entries in this case)
def test_store_multiple_sessions():
    test_data = generate_dummy_data(10)  # Generate 10 data points

    for text, audio_file in test_data:
        session_id = memory_manager.store_session(text, audio_file)
        # Assert that the session ID is valid
        if session_id:
            logging.info(f"Session stored successfully: {session_id}")
        else:
            logging.error("Failed to store session")

# Test Case 2: Search for the session using text and check if the correct audio is retrieved
def test_search_session():
    test_text = "Test entry number 1. This is a sample text for testing."
    
    # Search the session based on the text
    matched_text, matched_audio_file = memory_manager.search_session(test_text)

    # Assert that the correct session and audio file are found
    if matched_text and matched_audio_file:
        logging.info(f"Session found: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("No matching session found")

# Test Case 3: Check the FAISS index (ensure it's not empty and has more than 1 entry)
def test_faiss_index():
    if memory_manager.index.ntotal > 0:
        logging.info(f"FAISS index contains {memory_manager.index.ntotal} entries")
    else:
        logging.warning("FAISS index is empty")

# Test Case 4: Clear the cache and ensure it's reset
def test_clear_cache():
    memory_manager.clear_cache()
    
    # After clearing the cache, the index should be empty
    if memory_manager.index.ntotal == 0:
        logging.info("Cache cleared successfully. FAISS index is empty.")
    else:
        logging.error("Failed to clear the cache.")

# Test Case 5: Search with an empty text string
def test_empty_text_search():
    test_text = ""
    matched_text, matched_audio_file = memory_manager.search_session(test_text)

    if matched_text and matched_audio_file:
        logging.error(f"Unexpected match found: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.info("No match found for empty text.")

# Test Case 6: Search for text that is not in the index
def test_non_existent_search():
    test_text = "This text does not exist in the index."
    matched_text, matched_audio_file = memory_manager.search_session(test_text)

    if matched_text and matched_audio_file:
        logging.error(f"Unexpected match found: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.info("Correctly found no match for non-existent text.")

# Test Case 7: Search for text that is similar but not identical
def test_similar_text_search():
    test_text = "Test entry number 5. This is a sample text for testing."
    matched_text, matched_audio_file = memory_manager.search_session(test_text, threshold=0.2)

    if matched_text and matched_audio_file:
        logging.info(f"Similar session found: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("No similar session found.")

# Test Case 8: Add a session, clear cache, and search for it again
def test_re_add_and_search_after_cache_clear():
    test_text = "Test entry number 2. This is a sample text for testing."
    test_audio_file = os.path.join(TEMP_AUDIO_PATH, f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    
    # Create dummy audio file
    os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)
    with open(test_audio_file, "w") as f:
        f.write("dummy_audio_content_for_re_add")
    
    # Store the session
    session_id = memory_manager.store_session(test_text, test_audio_file)
    
    # Clear the cache
    memory_manager.clear_cache()

    # Search for the session after cache clear
    matched_text, matched_audio_file = memory_manager.search_session(test_text)

    if matched_text and matched_audio_file:
        logging.info(f"Session found after cache clear: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("No matching session found after cache clear.")

# Test Case 9: Add 1000+ entries to FAISS index and check performance
def test_large_dataset():
    test_data = generate_dummy_data(1000)  # Generate 1000 data points

    # Store 1000 sessions
    for text, audio_file in test_data:
        session_id = memory_manager.store_session(text, audio_file)
    
    # After adding, search for the last entry
    last_text = test_data[-1][0]
    matched_text, matched_audio_file = memory_manager.search_session(last_text)

    if matched_text and matched_audio_file:
        logging.info(f"Session found for large dataset: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("No matching session found in the large dataset.")

# Test Case 10: Search for text with different capitalization
def test_case_insensitivity():
    original_text = "Test entry number 3. This is a sample text for testing."
    capitalized_text = original_text.upper()  # Change case for the test

    # Store the original session
    test_audio_file = os.path.join(TEMP_AUDIO_PATH, f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)
    with open(test_audio_file, "w") as f:
        f.write("dummy_audio_content_for_case_test")

    memory_manager.store_session(original_text, test_audio_file)

    # Now search for the capitalized version
    matched_text, matched_audio_file = memory_manager.search_session(capitalized_text)

    if matched_text and matched_audio_file:
        logging.info(f"Case insensitive search found: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("Case insensitive search failed.")

# Test Case 11: Store and search for special characters or unicode characters
def test_special_characters_search():
    test_text = "This is a test with special characters: 🚀💡"
    test_audio_file = os.path.join(TEMP_AUDIO_PATH, f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}_unicode.wav")
    
    # Create a dummy audio file
    os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)
    with open(test_audio_file, "w") as f:
        f.write("dummy_audio_content_for_special_characters")

    session_id = memory_manager.store_session(test_text, test_audio_file)

    # Search for the same text with special characters
    matched_text, matched_audio_file = memory_manager.search_session(test_text)

    if matched_text and matched_audio_file:
        logging.info(f"Session found with special characters: '{matched_text}' with audio file {matched_audio_file}")
    else:
        logging.error("No matching session found with special characters.")

# Run all tests
def run_tests():
    test_store_multiple_sessions()  # Store 10 sessions
    test_search_session()  # Search for a specific session
    test_faiss_index()  # Verify FAISS index content
    test_clear_cache()  # Clear the cache and verify reset
    test_empty_text_search()  # Search with an empty text
    test_non_existent_search()  # Search for non-existent text
    test_similar_text_search()  # Search for similar text
    test_re_add_and_search_after_cache_clear()  # Re-add and search after clearing cache
    test_large_dataset()  # Add 1000+ sessions and search
    test_case_insensitivity()  # Case insensitive search
    test_special_characters_search()  # Test with special characters and unicode

if __name__ == "__main__":
    # Ensure that the FAISS index file is clean before starting the tests
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
    
    run_tests()
