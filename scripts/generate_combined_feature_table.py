#!/usr/bin/env python
# generate_combined_ASV_table.py
# Copyright Jackson M. Tsuji, 2024
# Creates a TSV-format QIIME2 feature table with overlaid taxonomy and sequence information for convenience of viewing

# Imports
import sys
import os
import time
import logging
import argparse
import re

from Bio.SeqIO.FastaIO import SimpleFastaParser
import pandas as pd

# GLOBAL VARIABLES
SCRIPT_VERSION = '0.9.0'

# Set up the logger
logging.basicConfig(format='[ %(asctime)s UTC ]: %(levelname)s: %(message)s')
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)


def load_feature_table(feature_table_filepath: str) -> pd.DataFrame:
    """
    Read a QIIME2 feature table

    :param feature_table_filepath: Path to the QIIME2 FeatureTable file, TSV format
    :return: QIIME2 FeatureTable[Frequency] artifact (pandas DataFrame)
    """
    # Load the table
    feature_table = pd.read_csv(feature_table_filepath, sep='\t', skiprows=1, header=0)
    
    # Check if the first column looks okay
    if feature_table.columns[0] != 'Feature ID':
        if feature_table.columns[0] == '#OTU ID':
            logger.debug('Renaming first column of feature table ("#OTU ID") to "Feature ID"')
            feature_table = feature_table.rename(columns={'#OTU ID': 'Feature ID'})
        elif 'Feature ID' in feature_table.columns:
            error = ValueError('"Feature ID" column already exists in provided feature table and is not the first '
                               'column.')
            logger.error(error)
            raise error
        else:
            error = RuntimeError('Do not recognize the first column of the feature table as feature IDs.')
            logger.error(error)
            raise error
            
    return feature_table


def normalize_feature_table(feature_table: pd.DataFrame, normalization_method='percent') -> pd.DataFrame:
    """
    Normalize a feature table (e.g., of counts) per sample.

    :param feature_table: QIIME2 FeatureTable[Frequency] artifact loaded as a pandas DataFrame
    :param normalization_method: method for normalization. Can choose 'percent' or 'proportion' currently.
    :return: QIIME2 FeatureTable[Frequency] artifact (pandas DataFrame) with normalized feature abundances
    """
    normalization_methods = ['percent', 'proportion']
    if normalization_method not in normalization_methods:
        error = ValueError(f'Input normalization method "{normalization_method}" is not one of the available '
                           f'normalization methods: "{",".join(normalization_methods)}".')
        logger.error(error)
        raise error

    # TODO: make a more generic way to deal with non-sample columns, given that other functions have similar code
    # TODO: consider masking these columns and then doing normalization, followed by unmasking.
    if ['Taxonomy', 'Sequence'] in feature_table.columns:
        error = RuntimeError('At least one of the "Taxonomy" or "Sequence" column names is present in the table. '
                             'Will not proceed with normalization.')
        logger.error(error)
        raise error

    logger.debug(f"Normalizing feature table using method: {normalization_method}")
    feature_table_normalized = feature_table.set_index('Feature ID')
    total_per_sample = feature_table.sum(axis=0)
    if normalization_method == 'proportion':
        feature_table_normalized = feature_table_normalized.div(total_per_sample)
    elif normalization_method == 'percent':
        feature_table_normalized = feature_table_normalized.div(total_per_sample).multiply(100)
    else:
        # This in theory should never be called, because the error should be caught above.
        raise ValueError()

    feature_table_normalized = feature_table_normalized.reset_index()

    return feature_table_normalized


def add_taxonomy_to_feature_table(feature_table: pd.DataFrame, taxonomy_filepath: str) -> pd.DataFrame:
    """
    Adds taxonomy values as the Taxonomy column to a QIIME2 feature table

    :param feature_table: QIIME2 FeatureTable[Frequency] artifact loaded as a pandas DataFrame
    :param taxonomy_filepath: Path to the taxonomy.tsv file output by the QIIME2 classifier
    :return: QIIME2 FeatureTable[Frequency] artifact with taxonomy in the Taxonomy column
    """
    # Check if Taxonomy column already exists
    if 'Taxonomy' in feature_table.columns:
        error = ValueError('"Taxonomy" column already exists in provided feature table. Cannot add taxonomy.')
        logger.error(error)
        raise error

    # Load taxonomy file
    logger.debug('Loading taxonomy file and adding taxonomy info')
    taxonomy_table = pd.read_csv(taxonomy_filepath, sep='\t')
    taxonomy_table = taxonomy_table.rename(columns={'Taxon': 'Taxonomy'})
    taxonomy_table = taxonomy_table[['Feature ID', 'Taxonomy']]
    
    # Merge
    feature_table = pd.merge(feature_table, taxonomy_table, how='left', on='Feature ID', validate='1:1')
    
    return feature_table


def add_sequences_to_feature_table(feature_table: pd.DataFrame, seq_filepath: str) -> pd.DataFrame:
    """
    Adds ASV/OTU sequences as the Sequence column to a QIIME2 feature table

    :param feature_table: QIIME2 FeatureTable[Frequency] artifact loaded as a pandas DataFrame
    :param seq_filepath: Path to the dna-sequences.fasta file output by the QIIME2 denoising/clustering step
    :return: QIIME2 FeatureTable[Frequency] artifact with representative sequences in the ReprSequences column
    """
    # Check if ReprSequence column already exists
    if 'Sequence' in feature_table.columns:
        error = ValueError('"Sequence" column already exists in provided feature table. Cannot add representative '
                           'sequences.')
        logger.error(error)
        raise error

    logger.debug('Loading and adding representative sequences')
    fasta_ids = []
    fasta_seqs = []
    with open(seq_filepath, 'r') as fasta_handle:
        for fasta_id, fasta_seq in SimpleFastaParser(fasta_handle):
            fasta_ids.append(fasta_id)
            fasta_seqs.append(fasta_seq)

    seq_table = pd.DataFrame({'Feature ID': fasta_ids, 'Sequence': fasta_seqs})
    
    # Merge
    feature_table = pd.merge(feature_table, seq_table, how='left', on='Feature ID', validate='1:1')
    
    return feature_table


def sort_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Roughly sorts a QIIME2 feature table by abundances of Features. The sort is performed based on the maximum
    count (or percent abundance) per feature.

    :param feature_table: QIIME2 FeatureTable[Frequency] artifact loaded as a pandas DataFrame
    :return: sorted QIIME2 FeatureTable[Frequency] artifact
    """
    # Choose a column name to store the sort info; name should not duplicate another column name in the table
    sort_column_id = 'sort'
    sort_column_id_checked = False
    while sort_column_id_checked is False:
        if sort_column_id in feature_table.columns:
            sort_column_id = sort_column_id + '-tmp'
        else:
            sort_column_id_checked = True

    # Remove non-sample columns other than Feature ID before sorting
    # A copy of the feature table is made just in case pandas changes the main table object when these edits are made
    feature_table_masked = feature_table.copy(deep=True)
    extraneous_non_sample_columns = ['Taxonomy', 'Sequence']
    for extraneous_non_sample_column in extraneous_non_sample_columns:
        if extraneous_non_sample_column in feature_table_masked.columns:
            logger.debug(f'Masking non-sample column "{extraneous_non_sample_column}" prior to sorting.')
            feature_table_masked = feature_table_masked.drop(columns=extraneous_non_sample_column)

    logger.debug('Sorting feature table by maximum count of each feature')
    max_value_per_feature = feature_table_masked.set_index('Feature ID').max(axis=1)
    feature_table[sort_column_id] = max_value_per_feature
    feature_table = feature_table.sort_values(by=[sort_column_id, 'Feature ID'], ascending=[False, True])
    feature_table = feature_table.drop(columns=sort_column_id)
    
    return feature_table


def parse_silva_taxonomy_entry(taxonomy_entry: str, resolve: bool = True) -> list:
    """
    Parse a single Silva taxonomy entry

    :param taxonomy_entry: Silva taxonomy string (see example below)
    :param resolve: Boolean of whether to resolve blank taxonomy entries with Unresolved_[last_classified_rank]
    :return: list of 7 rank entries for the taxonomy
    """
    # Example:
    # D_0__Bacteria;D_1__Margulisbacteria;D_2__microbial mat metagenome;D_3__microbial mat metagenome;
    # D_4__microbial mat metagenome;D_5__microbial mat metagenome;D_6__microbial mat metagenome

    taxonomy_split = str(taxonomy_entry).split(sep=';')

    if len(taxonomy_split) > 7:
        error = ValueError(f'Taxonomy entry is {len(taxonomy_split)} long, not 7 as expected. Full entry: '
                           f'"{taxonomy_entry}".')
        logger.error(error)
        raise error
    elif len(taxonomy_split) < 7:
        # Sometimes Silva entries are short; rest is unresolved
        for entry_index in range(len(taxonomy_split)+1, 8):
            taxonomy_split.append('')

    # Remove header pieces
    # TODO - confirm they are in the right order (0,1,2,3,4,5,6)
    taxonomy_split = [re.sub("D_[0-6]__", "", level) for level in taxonomy_split]

    # Fill in empty parts, if they exist
    if '' in taxonomy_split and resolve is True:
        taxonomy_split = resolve_taxonomy_entry(taxonomy_split)

    return taxonomy_split


def resolve_taxonomy_entry(taxonomy_split: list) -> list:
    """
    Fills in blank taxonomy entries in a list of taxonomy entries

    :param taxonomy_split: list of each taxonomy rank entry for a feature, with rank prefix removed. The list must be
                           sorted so that higher taxonomy entries (e.g., Domain) are before lower entries (e.g, Genus).
    :return: taxonomy_split but with empty entries filled in with Unresolved_[last_classified_rank]
    """
    # Get the positions of the blank entries
    empty_taxa = []
    for taxonomy_level in taxonomy_split:
        if taxonomy_level == '':
            empty_taxa.append(True)
        else:
            empty_taxa.append(False)

    # Get the numeric index of the first empty taxon
    # See https://stackoverflow.com/a/9542768, accessed Sept. 18, 2019
    first_empty_taxon = empty_taxa.index(True)

    if False in empty_taxa[first_empty_taxon:]:
        error = ValueError(f'There seems to be an empty entry in the middle of your taxonomy levels. Cannot resolve. '
                           f'Full taxonomy: "{",".join(taxonomy_split)}".')
        logger.error(error)
        raise error

    filler_entry = f'Unresolved_{taxonomy_split[(first_empty_taxon - 1)]}'
    for taxonomy_level_index in range(first_empty_taxon, 7):
        taxonomy_split[taxonomy_level_index] = filler_entry

    return taxonomy_split


def generate_combined_feature_table(feature_table_filepath: str, sequence_filepath: str, taxonomy_filepath: str,
                                    normalization_method: str = None, sort_features: bool = 'False',
                                    rename_features: bool = 'False', parse_taxonomy: bool = 'False',
                                    feature_id_colname: str = 'Feature ID') -> pd.DataFrame:
    """
    Loads and parses a feature table, along with optional sequence and taxonomy info, to generate a combined feature
    table. Optionally parses taxonomy, sorts features, and renames features.

    :param feature_table_filepath: path to the TSV-format feature-table.tsv output by QIIME2; a FeatureTable[Frequency]
                                   artifact exported as TSV.
    :param sequence_filepath: path to the dna-sequences.fasta output by QIIME2.
    :param taxonomy_filepath: path to the taxonomy.tsv output by QIIME2.
    :param normalization_method: set to 'percent' or 'proportion' to normalize the table to percent or proportion
                                 relative abundance, per sample. If None, no normalization is performed.
    :param sort_features: whether to sort features roughly based on abundances.
    :param rename_features: whether to rename features roughly based on abundance rank. Sort will also be performed.
    :param parse_taxonomy: whether to parse taxonomy into 7-rank taxonomy columns.
    :param feature_id_colname: name of the column where Feature IDs are stored in the output table.
    :return: a combined feature table as a pandas DataFrame.
    """
    # Set sort_features to True if rename_features is True
    if rename_features is True:
        logger.debug('Changing sort_features to True because rename_features is True.')
        sort_features = True

    # Load the feature table
    logger.debug('Loading feature table')
    feature_table = load_feature_table(feature_table_filepath)

    # Normalize
    if normalization_method is not None:
        feature_table = normalize_feature_table(feature_table, normalization_method)

    # Add taxonomy
    if taxonomy_filepath is not None:
        feature_table = add_taxonomy_to_feature_table(feature_table, taxonomy_filepath)

    # Parse taxonomy
    if parse_taxonomy is True:
        logger.info('Parsing taxonomy into 7 ranks')

        # TODO - expose 'resolve' option to user
        taxonomy_entries_parsed = map(lambda entry: parse_silva_taxonomy_entry(entry, resolve=True),
                                      feature_table['Taxonomy'].tolist())
        taxonomy_table_parsed = pd.DataFrame(taxonomy_entries_parsed,
                                             columns=['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus',
                                                      'Species'])

        # Bind to main table in place of 'Taxonomy'
        feature_table = pd.concat([feature_table, taxonomy_table_parsed], axis=1, sort=False)
        feature_table = feature_table.drop(columns='Taxonomy')

    # Add representative sequences
    if sequence_filepath is not None:
        feature_table = add_sequences_to_feature_table(feature_table, sequence_filepath)

    # Sort Feature IDs
    if sort_features is True:
        feature_table = sort_feature_table(feature_table)

    # Rename Feature IDs
    if rename_features is True:
        logger.info('Renaming feature IDs sequentially')
        num_rows = feature_table.shape[0]
        feature_table['Feature ID'] = range(num_rows)

    # Change first column to that desired by user
    if feature_id_colname != 'Feature ID':
        logger.info('Changing "Feature ID" colname to "' + feature_id_colname + '"')
        feature_table = feature_table.rename(columns={'Feature ID': feature_id_colname})

    return feature_table


def main(args):
    """
    Runs the workflow based on the provided command line arguments.
    """
    # Set logger verbosity
    if args.verbose is True:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Set the normalization method
    if args.percent is True:
        # TODO: consider exposing other normalization methods to the user
        normalization_method = 'percent'
    else:
        normalization_method = None

    # Set sort_features to True if rename_features is True
    if args.rename_features is True:
        args.sort_features = True

    # Check that taxonomy_filepath is set if parse_taxonomy is True
    if (args.parse_taxonomy is True) and (args.taxonomy is False):
        error = 'Although --parse_taxonomy was called, no --taxonomy_filepath was supplied.'
        logger.error(error)
        raise error

    # Startup messages
    logger.info(f'Running {os.path.basename(sys.argv[0])}')
    logger.info(f'Version: {SCRIPT_VERSION}')
    logger.debug('### SETTINGS ###')
    logger.debug(f'Feature table filepath: {args.feature_table}')
    logger.debug(f'Representative sequences filepath: {args.sequences}')
    logger.debug(f'Taxonomy filepath: {args.taxonomy}')
    logger.debug(f'Output table filepath: {args.output_feature_table}')
    logger.debug(f'Convert to percent relative abundances?: {args.percent}')
    logger.debug(f'Sort Feature IDs roughly by relative abundance?: {args.sort_features}')
    logger.debug(f'Rename Feature IDs sequentially?: {args.rename_features}')
    logger.debug(f'Parse Silva taxonomy into 7 ranks?: {args.parse_taxonomy}')
    logger.debug(f'Feature ID column name for the output table: {args.feature_id_colname}')
    logger.debug(f'Verbose logging: {args.verbose}')
    logger.debug('################')

    feature_table = generate_combined_feature_table(feature_table_filepath=args.feature_table,
                                                    sequence_filepath=args.sequences,
                                                    taxonomy_filepath=args.taxonomy,
                                                    normalization_method=normalization_method,
                                                    sort_features=args.sort_features,
                                                    rename_features=args.rename_features,
                                                    parse_taxonomy=args.parse_taxonomy,
                                                    feature_id_colname=args.feature_id_colname)
    # Write output
    if args.output_feature_table == '-':
        # Write to STDOUT
        logger.info("Writing merged table to STDOUT")
        feature_table.to_csv(sys.stdout, sep='\t', index=False)
    else:
        logger.info('Writing merged table to ' + args.output_feature_table)
        feature_table.to_csv(args.output_feature_table, sep='\t', index=False)

    logger.info(os.path.basename(sys.argv[0]) + ': done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a TSV-format QIIME2 feature table, with optional taxonomy '
                                                 'and sequence information. '
                                                 'Copyright Jackson M. Tsuji, 2024. '
                                                 f'Version: {SCRIPT_VERSION}')
    parser.add_argument('-f', '--feature_table', metavar='TSV', required=True,
                        help='The path to the input TSV feature table file.')
    parser.add_argument('-s', '--sequences', metavar='FASTA', required=False, default=None,
                        help='The path to the input FastA ASV/OTU sequence file. Sequences will be added as the '
                             '"Sequences" column. You can optionally omit this flag and not have sequences added to '
                             'the table.')
    parser.add_argument('-t', '--taxonomy', metavar='TSV', required=False, default=None,
                        help='The path to the input taxonomy file. Taxonomy will be added as the "Taxonomy" column. '
                             'You can optionally omit this flag and not have taxonomy added to the table.')
    parser.add_argument('-o', '--output_feature_table', metavar='TSV', required=False, default='-',
                        help='The path to the output TSV feature table. Will write to STDOUT (-) if nothing is '
                             'provided.')
    parser.add_argument('-p', '--percent', metavar='BOOL', required=False, action='store_true',
                        help='Optionally normalize to percent relative abundances for each sample.')
    parser.add_argument('-S', '--sort_features', metavar='BOOL', required=False, action='store_true',
                        help='Optionally sort Feature IDs roughly based on overall abundance.')
    parser.add_argument('-R', '--rename_features', metavar='BOOL', required=False, action='store_true',
                        help='Optionally rename the Feature IDs sequentially, roughly based on overall abundance. '
                             'Automatically sets --sort_features.')
    parser.add_argument('-P', '--parse_taxonomy', metavar='BOOL', required=False, action='store_true',
                        help='Optionally parse Silva taxonomy into 7 ranks with columns "Domain", "Phylum", etc.')
    parser.add_argument('-N', '--feature_id_colname', metavar='NAME', required=False,
                        default='Feature ID',
                        help='The name of the first column of the output feature table. [Default: "Feature ID"]')
    parser.add_argument('-v', '--verbose', metavar='BOOL', required=False, action='store_true',
                        help='Enable for verbose logging.')
    # TODO - add option to auto-detect if a QZA file is provided instead of the unpackaged file. Deal with the
    #  conversions. Same for if a BIOM file is provided.
    
    command_line_args = parser.parse_args()
    main(command_line_args)
