try:
    from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
    print("Imports successful")
    
    dg = DeepgramClient("test_key")
    print("Client initialized")
    
    conn = dg.listen.live.v("1")
    print("Connection object created")
    
    options = LiveOptions(model="nova-2")
    print("Options created")
    
except Exception as e:
    print(f"Error: {e}")
