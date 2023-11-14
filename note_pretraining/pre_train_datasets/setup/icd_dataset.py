import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class ICDDataset(object):

    def __init__(
            self,
            patient_id_column='PatientID',
            contact_date_column='ContactDTS',
            patient_encounter_id_column='PatientEncounterID',
            icd_code_column='CurrentICD10TXT',
            icd_sequence_column='icd_code_sequence',
            position_ids_column='position_ids',
            token_type_ids_column='token_type_ids',
            note_id_column='NoteID',
            note_text_column='NoteTXT'
    ):
        self._patient_id_column = patient_id_column
        self._contact_date_column = contact_date_column
        self._patient_encounter_id_column = patient_encounter_id_column
        self._icd_code_column = icd_code_column
        self._icd_sequence_column = icd_sequence_column
        self._position_ids_column = position_ids_column
        self._token_type_ids_column = token_type_ids_column
        self._note_id_column = note_id_column
        self._note_text_column = note_text_column

    def drop_duplicates(self, encounters_df):
        encounters_df.drop_duplicates(
            subset=[
                self._patient_id_column,
                self._contact_date_column,
                self._patient_encounter_id_column,
                self._icd_code_column
            ],
            inplace=True
        )

    def sort_dataframe(self, encounters_df):
        encounters_df.sort_values(
            [
                self._patient_id_column,
                self._contact_date_column,
                self._patient_encounter_id_column,
                self._icd_code_column
            ],
            ascending=True,
            inplace=True
        )

    def get_sequence_df(self, encounters_df):
        return (
            encounters_df
            .groupby(by=[self._patient_id_column, self._contact_date_column, self._patient_encounter_id_column])
            .agg({self._icd_code_column: lambda x: ' '.join(sorted(x))},)
            .rename(columns={self._icd_code_column: self._icd_sequence_column})
        )

    @staticmethod
    def check_subset_match(group):
        subset = dict()
        for sequence_1, sequence_2 in itertools.combinations(group, 2):
            if set(sequence_1.split()).issubset(sequence_2.split()):
                subset[sequence_1] = 1
            else:
                if subset.get(sequence_1, 0):
                    continue
                else:
                    subset[sequence_1] = 0
                    subset[sequence_2] = 0

        return [subset.get(sequence, 0) for sequence in group]

    def __get_partial_match_df(self, encounters_df):
        sequence_df = self.get_sequence_df(encounters_df=encounters_df)
        sequence_df['number_of_codes'] = sequence_df[self._icd_sequence_column].str.split().str.len()
        sequence_df = (sequence_df
        .reset_index()
        .drop_duplicates(
            [self._patient_id_column, self._contact_date_column, self._icd_sequence_column]
        ))
        # Compute sequences that have partial match
        sequence_df['subset_match'] = (
            sequence_df
            .sort_values('number_of_codes', ascending=True)
            .groupby(by=[self._patient_id_column, self._contact_date_column])[self._icd_sequence_column]
            .transform(self.check_subset_match)
        )
        return sequence_df


    def drop_partial_duplicates(self, encounters_df):
        sequence_df = self.__get_partial_match_df(encounters_df=encounters_df)
        # Drop partial subset matches
        sequence_df = sequence_df[~sequence_df['subset_match'].astype(bool)]
        # Return cleaned dataframe - The inner join will remove the partial matched rows
        return pd.merge(
            encounters_df,
            sequence_df[[self._patient_id_column, self._contact_date_column, self._patient_encounter_id_column]],
            on=[self._patient_id_column, self._contact_date_column, self._patient_encounter_id_column]
        )

    def __get_encounter_icd_counts_df(self, encounters_df):
        return encounters_df.groupby(self._patient_id_column).agg(
            encounter_count=pd.NamedAgg(column=self._patient_encounter_id_column, aggfunc=pd.Series.nunique),
            icd_count=pd.NamedAgg(column=self._icd_code_column, aggfunc=pd.Series.nunique)
        )

    def filter_by_number_encounters(self, encounters_df, cutoff=5):
        counts_df = self.__get_encounter_icd_counts_df(encounters_df)
        encounters_df = pd.merge(encounters_df, counts_df, on='PatientID')
        return encounters_df[(encounters_df['encounter_count'] >= cutoff)].reset_index(drop=True)

    def __get_cumulative_counts(self, encounters_df):
        ascending = (
            encounters_df
            .groupby([self._patient_id_column, self._contact_date_column])
            .agg(cumulative_count_ascending=(self._patient_encounter_id_column, 'count'))
            .groupby(level=0).cumsum()
        )

        descending = (
            encounters_df.sort_values(by=[self._patient_id_column, self._contact_date_column], ascending=False)
            .groupby([self._patient_id_column, self._contact_date_column], sort=False)
            .agg(cumulative_count_descending=(self._patient_encounter_id_column, 'count'))
            .groupby(level=0).cumsum()
        )

        return pd.merge(ascending, descending, on=[self._patient_id_column, self._contact_date_column])

    def get_df_with_sequences(self, encounters_df, sequence_df, sample_frac):
        if sample_frac is None:
            return encounters_df
        else:
            return pd.merge(
                encounters_df,
                (
                    sequence_df
                    .groupby([self._patient_id_column])
                    .sample(frac=sample_frac)[
                        [self._patient_id_column, self._contact_date_column, self._patient_encounter_id_column]
                    ]
                ),
                on=[self._patient_id_column, self._contact_date_column, self._patient_encounter_id_column]
            )

    def get_df_with_sequences_counts(self, encounters_df, sequence_df, sample_frac):
        df_with_sequences = self.get_df_with_sequences(
            encounters_df=encounters_df,
            sequence_df=sequence_df,
            sample_frac=sample_frac
        )
        return pd.merge(
            df_with_sequences,
            self.__get_cumulative_counts(df_with_sequences),
            on=[self._patient_id_column, self._contact_date_column]
        )

    def get_merged_sequence_df(self, encounters_df, sequence_df):

        sequence_df_merged = pd.merge(sequence_df, encounters_df, on=self._patient_id_column)

        # Create position and token type ids
        # Position ids indicate distance between notes in terms of days
        # Token type ids are used to indicate which icd codes correspond to the
        # current note. 1 - indicates they correspond to the current note.
        # 0 indicates that they are part of the patient history but not the current note
        sequence_df_merged[self._position_ids_column] = (
                sequence_df_merged[self._contact_date_column + '_y']
                - sequence_df_merged[self._contact_date_column + '_x']
        ).dt.days.astype(str)
        sequence_df_merged[self._token_type_ids_column] = (
                sequence_df_merged[self._patient_encounter_id_column + '_x']
                == sequence_df_merged[self._patient_encounter_id_column + '_y']
        ).astype(int).astype(str)

        return sequence_df_merged


    def __get_grouped_sequence_df(self, sequence_df_merged):
        return (
            sequence_df_merged
            .groupby([self._patient_id_column, self._note_id_column])
            .agg(
                token_type_counts=(self._token_type_ids_column, lambda x: sum([int(i) for i in x])),
                icd_code_sequence=(self._icd_code_column, ' '.join),
                position_ids=(self._position_ids_column, ' '.join),
                token_type_ids=(self._token_type_ids_column, ' '.join),
                patient_encounter_id=(self._patient_encounter_id_column + '_x', 'first'),
                note_text=(self._note_text_column, 'first'),
            )
            .reset_index()
        )

    def get_aggregated_sequence_df(self, encounters_df, sequence_df, max_seq_length, sample_number):
        sequence_df_merged = self.get_merged_sequence_df(
            encounters_df=encounters_df,
            sequence_df=sequence_df,
        )

        # Truncate long sequences - in both directions
        # Filter by sequence length - Truncate sequences longer this
        # Use the cumulative sum values to do this operation - ignore all rows
        # with values > 510. This essentially removes groups of dates such that
        # when we group by to get the icd sequences, we will have sequences < 510
        # and also ensure that we don't have partial sequences (i.e. half of the icd
        # codes in a note make it through - this could happen if you truncated the
        # sequence normally)
        sequence_df_ascending = (
            sequence_df_merged[sequence_df_merged['cumulative_count_ascending'] <= max_seq_length]
        )
        sequence_df_descending = (
            sequence_df_merged[sequence_df_merged['cumulative_count_descending'] <= max_seq_length]
        )

        sequence_df_grouped_ascending = self.__get_grouped_sequence_df(sequence_df_ascending)
        sequence_df_grouped_ascending = sequence_df_grouped_ascending[
            sequence_df_grouped_ascending['token_type_counts'] > 0
            ]
        sequence_df_grouped_descending = self.__get_grouped_sequence_df(sequence_df_descending)
        sequence_df_grouped_descending = sequence_df_grouped_descending[
            sequence_df_grouped_descending['token_type_counts'] > 0
            ]

        sequence_df_grouped = (
            pd.concat([sequence_df_grouped_ascending, sequence_df_grouped_descending])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if sample_number is None:
            return sequence_df_grouped
        else:
            return (
                sequence_df_grouped
                .groupby([self._patient_id_column])
                .sample(n=min(sequence_df_grouped.shape[0], sample_number))
            )

    @staticmethod
    def convert_parquet_to_json(sequence_df):
        for row in sequence_df.itertuples():

            icd_code_sequence = row.icd_code_sequence.split()
            position_ids = [int(position_id) for position_id in row.position_ids.split()]
            token_type_ids = [int(token_type_id) for token_type_id in row.token_type_ids.split()]

            if len(icd_code_sequence) != len(position_ids) or len(icd_code_sequence) != len(token_type_ids):
                raise ValueError('Sequence length mismatch')

            if not all(x <= y for x, y in zip(position_ids, position_ids[1:])):
                raise ValueError('Position ids are not increasing')

            if sum(token_type_ids) <= 0:
                print('Row has no token type ids')
                raise ValueError()

            if len(token_type_ids) > 510:
                raise ValueError('Sequence is longer than 510')

            if sum(np.array(position_ids)[np.array(token_type_ids, dtype=bool)]) != 0:
                raise ValueError('Position ids should be zero when token type ids are 1')

            yield {
                'patient_id': row.PatientID,
                'patient_encounter_id': str(row.patient_encounter_id),
                'note_id': row.NoteID,
                'text': row.note_text,
                'label_patient': row.icd_code_sequence,
                'label_note': ' '.join(
                    list(itertools.compress(row.icd_code_sequence.split(), [int(i) for i in row.token_type_ids.split()]))
                ),
                'position_ids': row.position_ids,
                'token_type_ids': row.token_type_ids
            }

    @staticmethod
    def drop_duplicate_codes_and_sort(codes):
        return ' '.join(sorted([*set(codes)]))


def run_transform():
    parser = ArgumentParser(
        description='Arguments provided at run time to create the tokenizer',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--encounters_file',
        type=str,
        required=True,
        help='The file that contains patient id, encounters etc'
    )
    parser.add_argument(
        '--notes_file',
        type=str,
        required=True,
        help='The file that contains patient id, encounters etc'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The file that contains patient id, encounters etc'
    )
    parser.add_argument(
        '--patient_id_column',
        type=str,
        default='PatientID',
        help=''
    )
    parser.add_argument(
        '--contact_date_column',
        type=str,
        default='ContactDTS',
        help=''
    )
    parser.add_argument(
        '--patient_encounter_id_column',
        type=str,
        default='PatientEncounterID',
        help=''
    )
    parser.add_argument(
        '--icd_code_column',
        type=str,
        default='CurrentICD10TXT',
        help=''
    )
    parser.add_argument(
        '--icd_sequence_column',
        type=str,
        default='icd_code_sequence',
        help=''
    )
    parser.add_argument(
        '--position_ids_column',
        type=str,
        default='position_ids',
        help=''
    )
    parser.add_argument(
        '--token_type_ids_column',
        type=str,
        default='token_type_ids',
        help=''
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=510,
        help=''
    )
    args = parser.parse_args()

    encounters_file = args.encounters_file
    notes_file = args.notes_file
    patient_id_column = args.patient_id_column
    contact_date_column = args.contact_date_column
    patient_encounter_id_column = args.patient_encounter_id_column
    icd_code_column = args.icd_code_column
    icd_sequence_column = args.icd_sequence_column
    max_seq_length = args.max_seq_length
    position_ids_column = args.position_ids_column
    token_type_ids_column = args.token_type_ids_column
    output_file = f'{Path(args.output_folder)}/{Path(encounters_file).stem}.json'

    encounters_df = pd.read_parquet(encounters_file)
    encounters_df = encounters_df[
        [patient_id_column, contact_date_column, patient_encounter_id_column, icd_code_column]
    ]

    icd_dataset = ICDDataset(
        patient_id_column=patient_id_column,
        contact_date_column=contact_date_column,
        patient_encounter_id_column=patient_encounter_id_column,
        icd_code_column=icd_code_column,
        icd_sequence_column=icd_sequence_column,
        position_ids_column=position_ids_column,
        token_type_ids_column=token_type_ids_column
    )

    print('Shape of original dataframe: ', encounters_df.shape)

    # Drop duplicates - Based on complete match
    icd_dataset.drop_duplicates(encounters_df=encounters_df)

    # Sort the dataframe by contact date
    icd_dataset.sort_dataframe(encounters_df=encounters_df)

    print('Shape after dropping duplicate Note ID and ICD code pairs: ', encounters_df.shape)

    # Drop partial duplicates - substring ICD code match
    encounters_df = icd_dataset.drop_partial_duplicates(encounters_df=encounters_df)

    print('Shape after partial duplicates: ', encounters_df.shape)

    encounters_df = icd_dataset.filter_by_number_encounters(encounters_df=encounters_df, cutoff=5)

    print('Shape after filtering by number of encounters: ', encounters_df.shape)

    sequence_df = pd.read_parquet(notes_file)[
        [patient_id_column, contact_date_column, patient_encounter_id_column, 'NoteID', 'ICD10CD', 'NoteTXT']
    ]
    sequence_df.rename({'ICD10CD': icd_sequence_column}, axis=1, inplace=True)
    sequence_df[icd_sequence_column] = (
        sequence_df[icd_sequence_column].str.split('|')
        .apply(icd_dataset.drop_duplicate_codes_and_sort)
    )

    print('Shape of sequence dataframe: ', sequence_df.shape)

    sample_fractions = [None]
    sample_number = [None]

    if len(sample_fractions) != len(sample_number):
        raise ValueError('The two lists should be of the same length')

    encounters_df_list = [
        icd_dataset.get_df_with_sequences_counts(
            encounters_df=encounters_df, sequence_df=sequence_df, sample_frac=sample_frac
        ) for sample_frac in sample_fractions
    ]

    print('Number of encounter dataframes: ', len(encounters_df_list))

    sequence_dataframes = [
        icd_dataset.get_aggregated_sequence_df(
            encounters_df=df,
            sequence_df=sequence_df,
            max_seq_length=max_seq_length,
            sample_number=number
        ) for df, number in zip(encounters_df_list, sample_number)
    ]

    print('Number of sequence dataframes: ', len(sequence_dataframes))

    with open(output_file, 'w') as file:
        for sequence_df in sequence_dataframes:
            for json_object in icd_dataset.convert_parquet_to_json(sequence_df):
                file.write(json.dumps(json_object) + '\n')

    print('Finished processing: ', str(Path(encounters_file).stem))


if __name__ == '__main__':
    run_transform()