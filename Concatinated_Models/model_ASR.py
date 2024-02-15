import whisper,time

class ASR_Model():
    
    def __init__(self,devices,model_type,audio_path,save_path,SAVE_ASR:bool):
        
        self.devices=devices          # # CPU or CUDA
        self.model_type=model_type    # You can use model_type with ("tiny","base","small","medium","large")
        self.audio_path=audio_path    # Path of Your Audio Data
        self.save_path=save_path      # Path that You Want to save Transformed text
        self.SAVE_ASR=SAVE_ASR        # if you want to save predicted text, you can Choose True 
        
    def Create_Model(self):
        
        
        model=whisper.load_model(self.model_type,device=self.devices)
        print("Model is created.")
        
        return model


    
    def Predicting(self,model):
        print("Model is predicting to text from audio...")
        
        start=time.time()
        result=model.transcribe(self.audio_path)
        stop=time.time()
        
        print(f"Model predicted to text from audio {(stop-start):.3f} Second")
        print("Generated Text: \n",result["text"])
        
        
        return result["text"]
        


    def Save_predicted(self,result):
        
        with open(self.save_path, 'w') as dosya:
        
            dosya.write(result)
            
    
    
    def forward(self):
        model=self.Create_Model()
        predict_text=self.Predicting(model=model)
        
        if self.SAVE_ASR==True:
            self.Save_predicted(result=predict_text)
        
        return predict_text
