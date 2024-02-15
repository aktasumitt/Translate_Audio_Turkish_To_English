import whisper,time

def Create_Model(devices,model_type:str="medium"):
    
    # You can use model_type with ("tiny","base","small","medium","large")
    model=whisper.load_model(model_type,device=devices)
    print("Model is created.")
    return model



def Predicting(model,audio_path):
    print("Model is predicting to text from audio...")
    
    start=time.time()
    result=model.transcribe(audio_path)
    stop=time.time()
    
    print(f"Model predicted to text from audio {(stop-start):.3f} Second")
    
    return result
    
    



def Save_predicted(result,save_path):
    
    with open(save_path, 'w') as dosya:
    
        dosya.write(result["text"])