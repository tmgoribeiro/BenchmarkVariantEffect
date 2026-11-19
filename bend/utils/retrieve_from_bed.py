'''
retrieve_from_bed.py
====================
Class to extract sequences from a reference genome using a `bed` file of genomic coordinates.

Example

``get_dna = Annotation(annotation = 'path/to/bed/file', reference_genome = '/path/to/genome/fasta')``

``get_dna.get_dna_segment(index = 0) # will return the dna segment for index 0 in the annotation file``

'''
from Bio import SeqIO
import pandas as pd



class Annotation:
    """An annotation object that can be used to retrieve DNA segments from a reference genome."""
    def __init__(self, annotation : str = None, 
                reference_genome : str = None):
        """
        Get an Annotation object that can retrieve sequences from a reference genome.

        Parameters
        ----------
        annotation : str, optional
            Path to a bed file containing genomic coordinates.
            The default is None.
        reference_genome : str, optional
            Path to a reference genome fasta file.
            The default is None.

        """
        # '''
        # Return a dna sequence from a bed file and a reference genome
        # annotation: path to bed file
        # reference_genome: path to reference genome fasta file
        # '''
        if annotation is not None:
            if isinstance(annotation, str):
                annotation = pd.read_csv(annotation, sep = '\t')

            self.annotation = annotation
        if reference_genome is not None: 
            self.genome_dict = SeqIO.to_dict(SeqIO.parse(reference_genome, "fasta"))

    
    def extend_segments(self, extra_context_left: int = None, extra_context_right: int = None, extra_context: int = None) -> None:
        # '''Modify the annotation to include extra context on both sides of the segments'''
        """
        Add extra context to the coordinates in the annotation file.
        Each sample in the annotation file will be extended by extra_context_left and extra_context_right.

        Parameters
        ----------
        extra_context_left : int, optional
            Number of nucleotides to add to the left of each segment.
            The default is None.
        extra_context_right : int, optional
            Number of nucleotides to add to the right of each segment.
            The default is None.
        extra_context : int, optional
            Number of nucleotides to add to both sides of each segment.
            Use this instead of extra_context_left and extra_context_right. 
            The default is None.

        Raises
        ------
        ValueError
            If extra_context is used simultaneously with extra_context_left or extra_context_right.

        Returns
        -------
        None.
        """

        if extra_context is not None:
            if extra_context_right is not None or extra_context_left is not None:
                raise ValueError('extra_context cannot be used with extra_context_left or extra_context_right')
            extra_context_left = extra_context
            extra_context_right = extra_context

        self.annotation.loc[:, 'start'] = self.annotation.loc[:, 'start'] - extra_context_left
        self.annotation.loc[:, 'end'] = self.annotation.loc[:, 'end'] + extra_context_right
            
    
    def get_item(self, index: int):
        """Get a row from the annotation file.

        Parameters
        ----------
        index : int
            Index of the row to return.

        Returns
        -------
        row : pandas.Row
            Row of the annotation file.
        """
        row = self.annotation.iloc[index] # return the row of the annotation

        return row

    def get_dna_segment(self, index) -> str:
        """Get a DNA sequence from the reference genome for a segment.

        Parameters
        ----------
        index : int
            Index of the row in the bed file for which to return the DNA sequence.
        
        Returns
        -------
        dna_segment : str
            The geomic DNA sequence of the segment.
        """


        
        item = self.get_item(index)

        # get dna segment from genome dict
        dna_segment =  str(self.genome_dict[item.chromosome].seq[int(item.start) : int(item.end)])

        return dna_segment

    def check_reference(self, variant_idx: int, max_mismatch_fraction: float = 0.05) -> int:
        """Check that the reference allele of each variant is found at ``variant_idx`` of its segment.

        A wrong genome build or coordinate convention raises no error anywhere else, it only gives
        wrong embeddings. Does nothing if the annotation has no ``ref`` column.

        Parameters
        ----------
        variant_idx : int
            Position of the variant inside each segment.
        max_mismatch_fraction : float, optional
            Fraction of mismatching variants above which an error is raised. The default is 0.05.

        Raises
        ------
        ValueError
            If more than ``max_mismatch_fraction`` of the reference alleles do not match the genome.

        Returns
        -------
        mismatches : int
            Number of variants whose reference allele does not match the genome.
        """
        if 'ref' not in self.annotation.columns:
            print('No ref column in the bed file, skipping the reference allele check')
            return 0

        mismatches = sum(
            self.get_dna_segment(index=i)[variant_idx].upper() != str(ref).upper()
            for i, ref in enumerate(self.annotation['ref'])
        )
        print(f'Reference allele check: {mismatches} of {len(self.annotation)} variants do not match the genome')
        if mismatches > max_mismatch_fraction * len(self.annotation):
            raise ValueError(f'More than {max_mismatch_fraction:.0%} of the reference alleles do not match the genome at the variant position. '
                             'Check the genome build and the coordinates of the bed file.')
        return mismatches 
    