from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
from typing import NoReturn, List


def get_vocab(icd_codes: pd.Series, cutoff: int) -> List[str]:
    """
    Get the vocabulary

    Args:
        icd_codes (pd.Series): A pandas series that contains the icd codes (one per row)
        cutoff (int): A frequency cutoff, codes occurring less than this number will not be part of the vocabulary

    Returns:
        (List[str]): A list that contains the tokens that make up the vocabulary.

    """
    icd_counts = icd_codes.value_counts()
    icd_counts = icd_counts[icd_counts >= cutoff]
    return list(icd_counts.index)


def run_create_vocab() -> NoReturn:
    """
    Create a tokenizer from the text present in the files.
    This tokenizer can then be used to tokenize texts
    """
    parser = ArgumentParser(
        description='Arguments provided at run time to create the tokenizer',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_file',
        required=True,
        type=str,
        help='The file that contains the icd codes'
    )
    parser.add_argument(
        '--icd_column_name',
        default='CurrentICD10TXT',
        type=str,
        help='The column in the file that contains the icd codes'
    )
    parser.add_argument(
        '--cutoff',
        default=0,
        type=int,
        help='The cutoff frequency to use when creating the vocab'
    )
    parser.add_argument(
        '--vocab_file',
        required=True,
        type=str,
        help='The file in which the icd vocab will be saved'
    )
    args = parser.parse_args()
    icd_codes = pd.read_parquet(args.input_file)[args.icd_column_name]
    icd_vocab = get_vocab(icd_codes=icd_codes, cutoff=args.cutoff)
    with open(args.vocab_file, 'w') as file:
        file.write(' '.join(icd_vocab))


if __name__ == '__main__':
    run_create_vocab()
