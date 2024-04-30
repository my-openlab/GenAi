After a breif study (3 days) I changed the design solution to be based on the following points:
choose a open source wake word model that is simple to implement and might work on resource constrained device. 
after a study with chatgpt (q & a) and looking around on the web, the one of the simplest and efficient model for RaspberryPi like devices 
is the [openwakeword](https://github.com/dscripka/openWakeWord). Although there is an more efficient [microwakeword](https://github.com/kahrendt/microWakeWord) model, openwakeword model is more direct and perhaps need less customizations. 

The model architecture is simple: 

openWakeword models are composed of three separate components:

  * melspectrogram : a 32 dimensional log-mel features from the provided audio samples using the following parameters:
        * stft window size: 25ms
        * stft window step: 10ms
        * mel band limits: 60Hz - 3800Hz
        * mel frequency bins: 32

        This is implemented as a dataflow graph as in ![melspectrogram model](images/melspectrogram.onnx.png). Although mathematically, the steps involves many more multiplications (fft -> abs -> pow2 -> log -> mel filterbank multiplications).
        A 1975msec audio recording, i.e. here results in (198,32) log mel embedding that can be fed into the next model. 


  * feature embeddings model: This model is provided by Google as a TFHub module under an Apache-2.0 license. For Genai, there is a python model that still being developed (using chatgpt, see model directory). The embedding model is developed based on the [paper](A https://arxiv.org/pdf/2005.06720). The referenced model is available at [kaggle](https://www.kaggle.com/models/google/speech-embedding/tensorFlow1/speech-embedding). "The architecture results in the first embedding requiring 12400 samples (775ms or 76 feature vectors). Each subsequent embedding vector requires a further 1280 samples (80ms or 8 feature vectors)". Thus resulting in expected embedding length = 16, when considering 1975msec audio recording, i.e. (198,32) input 

    Some of the details are not clearly understood but the initial model presented here closely matched to that being presented in the paper. The number of trainable parameters being 330K (329,929 to be exact). There could further tweaks to the model as the I get more time/roadblocks while developing. This is implemented as a dataflow graph as in ![embedding model](images/embedding_model.onnx.png). 

  * classification model: according the openWakeWord "The structure of this classification model is arbitrary, but in practice a simple fully-connected network or 2 layer RNN works well". For the time being, This part is already included in the feature embedding model. Ideally, this should have been something like ![alexa model](images/alexa_v0.1.onnx.png).

The above steps gives an intuition of the computational complexity. We can roughly estimate the number of multiplications needed. See queried prompts to aid this analysis[](model/design_thoughts.prompt.md)

| step   | calculation |  mults |  per time interval | input |output |
|----------|:-------------:|:------:|:------:|:------:|:------:|
| melspectrogram|  2×131,584×(round(n−257)/160​+1)+4,180,224+3n−1, n = 0.025*16000| 4708591 | 10ms |25 ms audio @ 16khz sampling rate |one 32-dimensional log-mel feature vector |
| embeddings model |    >= #trainable parameters   |   329929 | no calcs needed for first 775 ms,  then every 80ms upto 1975ms-775ms= 1200ms|76 log-mel feature vectors| one 96-dimensional embedded vector |
| keyword model | 128*1536 + (128*128) + 128 |    213120| no calcs needed for first 1975 ms, then every 80ms |16-embedded vector | 1 sigmoid output indicating yes/no|
|Total |                                        | 5251640 (~5.25M)  ||


Assume :
  - streaming mode i.e. continous audio, this means every 10ms, 5M multiplications have to be done. This is because when the last keyword model is run, there are no more melspectrogram windows to calculate. Of which the melspectrogram step needs max computations in shortest amount of time. Thus, making it to be an ideal candidate for acceleration.
  
  - user space clock frequency of 40 MHz

