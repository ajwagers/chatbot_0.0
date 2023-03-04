# chatbot_0.0 (Dadbot)
My first attempt at an AI chatbot, it is not intelligent and only tells dad jokes.  I am borrowing heavily from the links below, my ultimate goal is to add ASR (automatic speech recognition) and text-to-speech components.  I want the whole system to be self-contained and not dependent on an internet connection or externally managed API.

How to run the "Dadbot"
- Navigate to the folder containing the files in this repository.
- If you are using Anaconda, activate it.
    ```bash
      conda activate
    ```
- Train the model.
    ```bash
      python train.py
    ```
- Then you can run the model.
    ```bash
      python chatbot_0_0.py
    ```

https://github.com/patrickloeber/pytorch-chatbot
https://www.python-engineer.com/posts/chatbot-pytorch/
https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
https://github.com/UsmanNiazi/Chatbot-with-Pytorch
