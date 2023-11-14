import re
import ftfy
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main():
    cli_parser = ArgumentParser(
        description='configuration arguments provided at run time from the CLI',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='The folder that contains the data'
    )
    cli_parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='The split of the data to process'
    )
    cli_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='The file where we save the cleaned note texts'
    )
    cli_parser.add_argument(
        '--note_id_column',
        type=str,
        default='NoteID',
        help='The column that contains the Note IDS'
    )
    cli_parser.add_argument(
        '--note_text_column',
        type=str,
        default='NoteTXT',
        help='The column that contains the Note text'
    )
    cli_parser.add_argument(
        '--unicode_fix',
        action='store_true',
        help='Fix the unicode text using the ftfy library'
    )
    cli_parser.add_argument(
        '--long_char_fix',
        action='store_true',
        help='Shorten long sequences of special characters'
    )
    cli_parser.add_argument(
        '--long_char_max',
        type=int,
        default=8,
        help='Sequences longer than this are considered long and are shortened to this length'
    )
    cli_parser.add_argument(
        '--token_length_filter',
        type=int,
        default=300,
        help='Sequences shorter than this are filtered out'
    )

    args = cli_parser.parse_args()

    data_frame = pd.concat(
        [pd.read_parquet(x) for x in tqdm(Path(args.input_folder).glob(f'{args.split}_notes*.parquet'))]
    )

    text_filter = r'(^.{0,200}(case(\W*|.{0,20})management))|(^.{0,200}(social(\W*|.{0,20})work(\W*|.{0,' \
                  r'20})progress))|(^.{0,200}(spiritual(\W*|.{0,20})care))'

    print('Drop duplicate note ids')
    data_frame.drop_duplicates(subset=args.note_id_column, inplace=True)

    print('Filtering based on note text and note length')
    data_frame = data_frame[
        (data_frame[args.note_text_column].str.split().str.len() >= args.token_length_filter) &
        ~(data_frame[args.note_text_column].str.contains(text_filter, regex=True, case=False))
    ]

    data_frame.reset_index(drop=True, inplace=True)

    if args.unicode_fix:
        print('Fixing unicode')
        data_frame[args.note_text_column] = data_frame[args.note_text_column].apply(ftfy.fix_text)

    if args.long_char_fix:
        print('Long character fix')
        data_frame[args.note_text_column] = data_frame[args.note_text_column].apply(
            lambda x: re.sub(
                rf'(?P<specchar>([^a-zA-Z0-9_\s]|_)){{{args.long_char_max},}}', r'\g<specchar>' * args.long_char_max, x)
        )
    data_frame.to_parquet(args.output_file)


if __name__ == '__main__':

    main()
