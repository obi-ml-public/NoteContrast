import random
import tempfile
from typing import NoReturn, List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import RobertaProcessing, BertProcessing
from transformers import PreTrainedTokenizerFast


class CreateTokenizer(object):

    def __init__(self, files: List[str], special_tokens: List[str]) -> NoReturn:
        """
        Create a tokenizer from the text present in the files.
        This tokenizer can then be used to tokenize texts.

        Args:
            files (List[str]): A list of files that contains the text from which to construct the tokenizer/vocabulary
            special_tokens (List[str]): A list of special tokens to add to the vocabulary

        """
        self.__tokenizer = Tokenizer(WordLevel())
        self.__tokenizer.pre_tokenizer = Split(' ', behavior='removed')
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=100000)
        self.__tokenizer.enable_truncation(max_length=512, direction='left')
        self.__tokenizer.train_from_iterator(self.get_corpus(files), trainer)
        self.__tokenizer.post_processor = RobertaProcessing(
            cls=('<s>', self.__tokenizer.token_to_id('<s>')),
            sep=('</s>', self.__tokenizer.token_to_id('</s>')), trim_offsets=False, add_prefix_space=False
        )
        # Use if using BERT tokenizer - Comment out the previous
        # section of code before using
        # self.__tokenizer.post_processor = BertProcessing(
        #     cls=('[CLS]', self.__tokenizer.token_to_id('[CLS]')),
        #     sep=('[SEP]', self.__tokenizer.token_to_id('[SEP]'))
        # )

    @staticmethod
    def get_corpus(files):
        if not isinstance(files, list):
            files = [files]
        for file in files:
            for line in open(file):
                yield line.strip()

    def print_vocab_details(self, size: int = 30) -> NoReturn:
        """
        Print the size and the desired amount of elements from the vocabulary

        Args:
            size (int, defaults to `30`): How many elements from the vocabulary to print

        """
        print('Size of the vocabulary: ', self.__tokenizer.get_vocab_size())
        print(
            f'Showing {size} elements of vocabulary: ',
            dict(random.sample(sorted(self.__tokenizer.get_vocab().items()), size))
        )

    def get_tokenizer(self) -> Tokenizer:
        """
        Return the tokenizer object

        Returns:
            (Tokenizer): The tokenizer object

        """
        return self.__tokenizer

    def save_tokenizer(self, save_file: str) -> NoReturn:
        """
        Save the tokenizer to a file. This tokenizer file can be used
        to load the tokenizer and tokenize your datasets.

        Args:
            save_file (str): The file in which the tokenizer model is going to be saved

        """
        self.__tokenizer.save(save_file)

    def get_hf_tokenizer(
            self,
            name_or_path,
            model_max_length,
            truncation_side='right',
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
    ):
        temp_file = tempfile.NamedTemporaryFile()
        self.__tokenizer.save(temp_file.name)
        return PreTrainedTokenizerFast(
            name_or_path=name_or_path,
            tokenizer_file=temp_file.name,
            truncation_side=truncation_side,
            model_max_length=model_max_length,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token='<mask>',
        )


def run_create_tokenizer() -> NoReturn:
    """
    Create a tokenizer from the text present in the files.
    This tokenizer can then be used to tokenize texts
    """
    parser = ArgumentParser(
        description='Arguments provided at run time to create the tokenizer',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='The files that contain text from which we build our tokenizer'
    )
    parser.add_argument(
        '--special_tokens',
        nargs='+',
        default=['<s>', '<pad>', '</s>', '<unk>', '<mask>'],
        help='The special tokens - for example start, end, padding and mask tokens'
    )
    parser.add_argument(
        '--save_file',
        required=True,
        type=str,
        help='The file in which the tokenizer model is going to be saved'
    )
    args = parser.parse_args()
    create_tokenizer = CreateTokenizer(
        files=args.files,
        special_tokens=args.special_tokens
    )
    create_tokenizer.print_vocab_details()
    tokenizer = create_tokenizer.get_hf_tokenizer(
        name_or_path='icd_roberta',
        model_max_length=512
    )
    tokenizer.save_pretrained(args.save_file)

    #create_tokenizer.save_tokenizer(save_file=args.save_file)


if __name__ == '__main__':
    run_create_tokenizer()
