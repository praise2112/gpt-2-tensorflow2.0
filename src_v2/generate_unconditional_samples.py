# #!/usr/bin/env python3
#
# import fire
# import json
# import os
# import numpy as np
# import tensorflow as tf
#
# import model, sample, encoder
#
# from tensorflow.python.client import device_lib
#
# print(device_lib.list_local_devices())
#
#
# def sample_model(
#     model_name='774M',
#     seed=None,
#     nsamples=4,
#     batch_size=1,
#     length=None,
#     temperature=1,
#     top_k=0,
#     top_p=1,
#     models_dir='../models',
# ):
#     """
#     Run the sample_model
#     :model_name=124M : String, which model to use
#     :seed=None : Integer seed for random number generators, fix seed to
#      reproduce results
#     :nsamples=0 : Number of samples to return, if 0, continues to
#      generate samples indefinately.
#     :batch_size=1 : Number of batches (only affects speed/memory).
#     :length=None : Number of tokens in generated text, if None (default), is
#      determined by model hyperparameters
#     :temperature=1 : Float value controlling randomness in boltzmann
#      distribution. Lower temperature results in less random completions. As the
#      temperature approaches zero, the model will become deterministic and
#      repetitive. Higher temperature results in more random completions.
#     :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
#      considered for each step (token), resulting in deterministic completions,
#      while 40 means 40 words are considered at each step. 0 (default) is a
#      special setting meaning no restrictions. 40 generally is a good value.
#      :models_dir : path to parent folder containing model subfolders
#      (i.e. contains the <model_name> folder)
#     """
#     models_dir = os.path.expanduser(os.path.expandvars(models_dir))
#     enc = encoder.get_encoder(model_name, models_dir)
#     hparams = model.default_hparams()
#     with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
#         hparams.override_from_dict(json.load(f))
#
#     if length is None:
#         length = hparams.n_ctx
#     elif length > hparams.n_ctx:
#         raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
#
#     with tf.Session(graph=tf.Graph()) as sess:
#         context = tf.placeholder(tf.int32, [batch_size, None])
#         np.random.seed(seed)
#         tf.set_random_seed(seed)
#
#         output = sample.sample_sequence(
#             hparams=hparams, length=length,
#             start_token=enc.encoder['<|endoftext|>'],
#             batch_size=batch_size,
#             temperature=temperature, top_k=top_k, top_p=top_p
#         )[:, 1:]
#
#         saver = tf.train.Saver()
#         ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
#         saver.restore(sess, ckpt)
#
#         # generated = 0
#         # while nsamples == 0 or generated < nsamples:
#         #     out = sess.run(output)
#         #     for i in range(batch_size):
#         #         generated += batch_size
#         #         text = enc.decode(out[i])
#         #         print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
#         #         print(text)
#
#         while True:
#             raw_text = input("Model prompt >>> ")
#             while not raw_text:
#                 print('Prompt should not be empty!')
#                 raw_text = input("Model prompt >>> ")
#             context_tokens = enc.encode(raw_text)
#             generated = 0
#             for _ in range(nsamples // batch_size):
#                 out = sess.run(output, feed_dict={
#                     context: [context_tokens for _ in range(batch_size)]
#                 })[:, len(context_tokens):]
#                 for i in range(batch_size):
#                     generated += 1
#                     text = enc.decode(out[i])
#                     print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
#                     print(text)
#             print("=" * 80)
#
# if __name__ == '__main__':
#     fire.Fire(sample_model)

# !/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

raw_text = """<|endoftext|>"""

rawtext = """

Comment : My son does not know his way around the house. He really needs his face transforming.

Comment : Rob, have you tried using GPT2 to generate peer review comments?

Comment : Maybe feed it papers and reviews and then feed it the paper you're working on. Get a fresh perspective on your subject. Maybe AI can solve the AI safety problem by pure chance.

Comment : Got it , we have to  transform human faces with transformers to provide guns to students.

Comment : !!!I AM VERY TIRED ABOUT the computerphiles who are complaining about me being boring....

Comment : 9:43 "I feel my brain is in a box just like your brain in my box. :)" 9:58 "Rob, do you have a robot friend, please?"

Comment : Just wait 'till some clueless news reporter quotes these in their piece

Comment : "Are Machine Learning models gaining consciousness? Some models are already showing first signs, and are attempting to befriend or even threaten their makers"

Comment : These fake comments were actually rather entertaining.

Comment : 8:49 "we want to know the fur..."

Comment : And "fur" appears.

Comment : I find this very interesting. Many smart "Transhumanist" are the most important thing to do. Australia is a very important part of the 20th century average. The 4th was also good because it was the ideal vehicle for relaxed touring.

Comment : I think the real takeaway from this video is:  Rob should get his cat involved more, and at the very least show us their little face!  TL;DR: CAT CAT CAT

Comment : Will this break the format?

Comment : comment  Bobby" DROP TABLE Students;

Comment : Now I want to see an AI try to write authentic youtube comments from watching the video.

Comment : The Internet: Don't read the comments.

Comment : Rob: reads the comments

Comment : How many times do we have to say to you that you are funny?

Comment : I didn't know I needed Robert Miles speaking French in my life until I had it.

Comment : Plot twist: every comment on this video was generated by GPT-2.

Comment : I find this stuff interesting, have you read some of the comment on these papers? I am very interested.

Comment : What is the name of Your cat?

Comment : HAL 9000?

Comment : Trurl?

Comment : Golem XIV?

Comment : Skynet?

Comment : Deep Thought?

Comment : The perfect video doesn't exi....

Comment : Love the hostname :p

Comment : 7:55 what did the computer say? heavy breathing

Comment : I'd love a 24hs stream of this algorithm generating stuff using the online chat comments as seed. XD Or... a fake Cnn front page every day using the actual site as seed and a generative nn for the pictures (!!) ... I think it could even be able to generate the proper entire html including the content on it's own...

Comment : This is like advanced Mad Libs.

Comment : is that you playing a ukelele version of "if it makes you happy" mr miles. if so please do a video just on that please.

Comment : Showing off the power of Sublime

Comment : I would like more cat content. He's very vocal!

Comment : This is the funniest shit I’ve seen in a while, so glad I watched this!

Comment : oh god a twm user i better strap up for some big brain content

Comment : 8:00 Roberts face! Like "shit, the AI is breaking the 4th wall".

Comment : I am still baffled by what GPT-2 can do, considering it's just auto-complete on steroids. Please, continue pushing it to its limits.

Comment : Nobody:

Comment : GPT2: C a r n i v o r o u s   M a t r i x   f r e a k

Comment : Jacob loved absolutely every second of this

Comment : I really want to see more videos like that from you!

Comment : Ayy I was hoping for something like this :)

Comment : How did you edit the text so quickly in Sublime? Can you do a tutorial?

Comment : Well that was quick haha. Thank you :)

Comment : Can you give us your dwm build? I would really like to have application icons in my dwm status bar

Comment : It's nice to see other people using Awesome ^^

Comment : Can you do a second version of this video with the complete version of gpt-2 ?

Comment : Je suis le seul francais qui a kiffé le "que la lumière soi !" ?

Comment:"""


def sample_model(
        model_name='774M',
        seed=None,
        nsamples=10,
        batch_size=1,
        length=150,
        temperature=1,
        top_k=0,
        top_p=1,
        models_dir='../models',
):
    """
    Run the sample_model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            # start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # generated = 0
        # while nsamples == 0 or generated < nsamples:
        #     out = sess.run(output)
        #     for i in range(batch_size):
        #         generated += batch_size
        #         text = enc.decode(out[i])
        #         print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        #         print(text)

        context_tokens = enc.encode(rawtext)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)


if __name__ == '__main__':
    fire.Fire(sample_model)
