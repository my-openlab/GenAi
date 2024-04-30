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

    * feature embeddings model: This model is provided by Google as a TFHub module under an Apache-2.0 license. For Genai, there is a python model that 
    still being developed (using chatgpt, see model directory). The embedding model is developed based on the [paper](A https://arxiv.org/pdf/2005.06720).
    Some of the details are not clearly understood but the initial model presented here closely matched to that being presented in the paper. The number of trainable parameters being 330K (329,929 to be exact). There could further tweaks to the model as the I get more time/roadblocks while developing. 

    * classification model: according the openWakeWord "The structure of this classification model is arbitrary, but in practice a simple fully-connected network or 2 layer RNN works well". For the time being, This model is ini



