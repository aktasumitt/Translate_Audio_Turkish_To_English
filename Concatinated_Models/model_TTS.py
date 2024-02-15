import torchaudio
import torch
from scipy.io.wavfile import write
import tqdm
import matplotlib.pyplot as plt


class TEXT_TO_SPEECH():
    
    "We Will Use Tacotron and WaveGlow Model in this class"

    
    def __init__(self,devices,Text,VISUALIZE_WAVE_AND_SPEC:bool,SAVE_AUDIO_PATH:str) :
        
        
        self.VISUALIZE_WAVE_AND_SPEC=VISUALIZE_WAVE_AND_SPEC    # You Can Choose True if You Want to Visulize
        self.devices=devices                                    # Cuda or cpu
        self.Text=Text                                          # Text That Want to Transform Speech
        self.SAVE_AUDIO_PATH=SAVE_AUDIO_PATH                    # Path where you want to save
    
    
    
    def Create_Models(self):
        print("Creating Models...")
        
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        processor = bundle.get_text_processor()
        tacotron2 = bundle.get_tacotron2().to(self.devices)
        vocoder = bundle.get_vocoder().to(self.devices)
            
        return processor,tacotron2,vocoder


    
    
    def Seperate_Text(self):

        print("\n\nCreate text list from your text file...")
        text_list=[]
        self.Text[0]=self.Text[0].replace(":",".")
        self.Text[0]=self.Text[0].replace(";",".")
        
        for i in self.Text[0].split("."):
            text_list.append(i)
        
        return text_list[:-1]



    
    def Tacotron_processor(self,text_list,processor,tacotron2):     
        
        spec_list=[]
        spec_lengths_list=[]
        print("\nTacotron is predicting texts...")
        
        progrss_bar=tqdm.tqdm(range(len(text_list)),"Training_progress",position=0,leave=True)
        
        for text in text_list:    
            with torch.inference_mode():
                processed, lengths = processor(text)
                processed = processed.to(self.devices)
                lengths = lengths.to(self.devices)
                spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
                spec_lengths_list.append(spec_lengths)
                spec_list.append(spec)
                
                progrss_bar.update(1)
        
        progrss_bar.close()
        print("\nPredicting is finished...")
        
        return spec_list,spec_lengths_list



    
    def WaveGlow_Vocoder(self,spec_list,spec_lengths_list,vocoder):
        
        waveform_list=[]
        lengths_list=[]
        
        print("\nWaveGlow is predicting texts...")
        progrss_bar=tqdm.tqdm(range(len(spec_list)),"Training_progress",position=0,leave=True)

        
        for i in range(len(spec_list)):
            waveforms, lengths = vocoder(spec_list[i], spec_lengths_list[i])
            waveform_list.append(waveforms)
            lengths_list.append(lengths)
        
            progrss_bar.update(1)
        
        progrss_bar.close()
        print("\nPredicting is finished")
        
        return waveform_list,lengths_list




    def Plot_Waveform_and_Spec(self,waveforms_list, spec_list):
        
        for i,waveforms in enumerate(waveforms_list,0):
            
            waveforms = waveforms.cpu().detach()

            fig, [ax1, ax2] = plt.subplots(2, 1)
            ax1.plot(waveforms[0])
            ax1.set_xlim(0, waveforms.size(-1))
            ax1.grid(True)
            ax2.imshow(spec_list[i][0].cpu().detach(), origin="lower", aspect="auto")
            plt.show()

    
    
    def Saving_Audio(self,waveforms_list,sample_rate,saving_path):
            
        concat_waveform = waveforms_list[0]

        for waveform in waveforms_list[1:]:
            concat_waveform = torch.cat((concat_waveform, waveform), dim=1)

        concat_waveform = concat_waveform.cpu().numpy()   
        write((f"{saving_path}"+f"\\test.wav"), sample_rate, concat_waveform[0])  
        
        print("\nSaving is finished")
        
    
    
    def forward(self):
        
        # Create Models
        processor,tacotron2,vocoder=self.Create_Models()


        # We seperated text for the better speech
        seperated_text=self.Seperate_Text()



        # Training Tacotron for Creating Spectrogram from Texts
        spec_list,spec_lengths_list=self.Tacotron_processor(text_list=seperated_text,
                                                            processor=processor,
                                                            tacotron2=tacotron2)


        # Training WaveGlow for Creating Waveform from Spectogram
        waveform_list,lengths_list=self.WaveGlow_Vocoder(spec_list=spec_list,
                                                        spec_lengths_list=spec_lengths_list,
                                                        vocoder=vocoder)

        

        # Visualize Waveform and spectogram from prediction.
        if self.VISUALIZE_WAVE_AND_SPEC==True:
            self.Plot_Waveform_and_Spec(waveforms_list=waveform_list,spec_list=spec_list)


        # Save and listen Audio.
        audios=self.Saving_Audio(waveforms_list=waveform_list,
                                sample_rate=vocoder.sample_rate,
                                saving_path=self.SAVE_AUDIO_PATH)
                        
        
        print("RESULT AUDIO SAVED Predicted Folder")
        
        return audios
            