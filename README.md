# homo-generator project overview
This project uses the the [Homosaurus
vocabulary](https://homosaurus.org/), an international linked data
vocabulary of LBGTQ+ terms, to finetune a text generation model.

The base model (the model used as a foundation for finetuning) comes
EleutherAI's
[gpt-neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m) model.
You can see [the code](./train/train.py) that I used to fine-tune the
model.

