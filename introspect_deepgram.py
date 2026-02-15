import deepgram
print(dir(deepgram))
try:
    from deepgram import DeepgramClient
    print("DeepgramClient found")
except ImportError:
    print("DeepgramClient NOT found")

try:
    import deepgram.clients
    print(dir(deepgram.clients))
except ImportError:
    print("deepgram.clients import failed")
