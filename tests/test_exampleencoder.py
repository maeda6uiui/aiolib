import logging
import sys
import os
sys.path.append(os.path.abspath("../src/aiolib"))

from util.encode import ExampleEncoder

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)

if __name__=="__main__":
    train_example_filepath="./WorkingDir/Data/train_questions.json"
    dev1_example_filepath="./WorkingDir/Data/dev1_questions.json"
    dev2_example_filepath="./WorkingDir/Data/dev2_questions.json"
    leaderboard_example_filepath="./WorkingDir/Data/aio_leaderboard.json"
    context_filepath="./WorkingDir/Data/candidate_entities.json.gz"
    bert_model_dir="USE_DEFAULT"

    train_save_dir="./WorkingDir/EncodedText/Train"
    dev1_save_dir="./WorkingDir/EncodedText/Dev1"
    dev2_save_dir="./WorkingDir/EncodedText/Dev2"
    leaderboard_save_dir="./WorkingDir/EncodedText/Leaderboard"

    encoder=ExampleEncoder(context_filepath,bert_model_dir)
    #encoder.encode_save(train_example_filepath,train_save_dir)
    encoder.encode_save(dev1_example_filepath,dev1_save_dir)
    #encoder.encode_save(dev2_example_filepath,dev2_save_dir)
    #encoder.encode_save(leaderboard_example_filepath,leaderboard_save_dir)
