import torch,models,dataset,train,config,visualization

if __name__=="__main__":
    
    
    
    devices = "cuda:0" if torch.cuda.is_available() else "cpu"



    # Create Models
    processor,tacotron2,vocoder=models.Create_Bundles(devices=devices)



    # Create Text_List.
    # If you want give input manually, you can use want_input=True or  You can give string text
    Text_list=dataset.Create_Text(want_input=False,
                                want_text=True,
                                text=config.TEXT)

    print("Seperated Text: \n",Text_list)

    # Training Tacotron for Creating Spectrogram from Texts
    spec_list,spec_lengths_list=train.Tacotron_processor(text_list=Text_list,
                                                        devices=devices,
                                                        processor=processor,
                                                        tacotron2=tacotron2)



    # U will be able to visualize spectogram with waveform after wave_glow_vocoder 
    if config.VISUALIZE_SPEC==True:
        visualization.plot_spectogram(spec_list=spec_list)




    # Training WaveGlow for Creating Waveform from Spectogram
    waveform_list,lengths_list=train.WaveGlow_Vocoder(spec_list=spec_list,
                                                    spec_lengths_list=spec_lengths_list,
                                                    vocoder=vocoder)

    

    # Visualize Waveform and spectogram from prediction.
    if config.VISUALIZE_WAVE_AND_SPEC==True:
        visualization.Plot_Waveform_and_Spec(waveforms_list=waveform_list,spec_list=spec_list)
        


    # Save and listen Audio.
    audios=train.Saving_Audio(waveforms_list=waveform_list,
                            sample_rate=vocoder.sample_rate,
                            saving_path=config.SAVE_AUDIO_PATH)
                    

        







