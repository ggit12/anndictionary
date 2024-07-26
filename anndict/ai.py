#AI integration for cell typing, interpretation of gene lists, and other labelling tasks
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import subprocess

from openai import OpenAI

def set_openai_api_key(api_key):
    """
    Sets the OpenAI API key as an environment variable.
    
    Args:
        api_key (str): The OpenAI API key.
    """
    os.environ['OPENAI_API_KEY'] = api_key

def get_openai_client():
    """
    Retrieves the OpenAI API key from the environment variable and initializes the OpenAI client.

    Returns:
        OpenAI: An instance of the OpenAI client initialized with the API key.

    Raises:
        ValueError: If the OpenAI API key is not set in the environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable. This can be done by running `export OPENAI_API_KEY='your-openai-api-key'` in terminal or using anndict.ai.set_openai_api_key('my-api-key')")
    
    return OpenAI(api_key=api_key)

def enforce_semantic_list(lst):
    error_message = "gene_list appears to contain numeric or numeric cast as string. Please ensure you are passing semantic labels (i.e. gene symbols or cell types) and not integer labels for AI interpretation."
    
    if not all(isinstance(item, str) for item in lst):
        raise ValueError(error_message)
    
    try:
        [float(item) for item in lst]
        raise ValueError(error_message)
    except ValueError:
        return True

def attempt_ai_integration(ai_func, fallback_func, *args, **kwargs):
    """
    Attempts to run the AI integration function with the provided arguments. If an exception occurs,
    runs the fallback function instead.

    Args:
        ai_func (callable): The AI integration function to run.
        fallback_func (callable): The fallback function to run if the AI function fails.
        args: Variable length argument list for the AI function.
        kwargs: Arbitrary keyword arguments for the AI function.

    Returns:
        The result of the AI function if it succeeds, otherwise the result of the fallback function.
    """
    try:
        return ai_func(*args, **kwargs)
    except Exception as e:
        print(f"AI integration failed: {e}")
        return fallback_func()

def generate_file_key(file_path):
    """
    Generates a concise, descriptive name for a file using the OpenAI Chat Completions API based on its file path.

    Args:
        file_path (str): The path to the file for which to generate a name.

    Returns:
        str: A generated name for the file that can be used as a dictionary key.

    Example:
        >>> file_key = generate_file_key('example.txt')
        >>> print(file_key)
        'ExampleFileName'
    """
    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a python dictionary key generator. Examples: /Users/john/data/single_cell_liver_data.h5ad -> liver, /Users/john/data/single_cell_heart_normalized.h5ad -> heart_normalized"},
        {"role": "user", "content": f"{file_path} -> "}
    ]

    # Initialize the OpenAI client
    client = get_openai_client()

    # Call the OpenAI Chat Completions API to generate a name
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=10,  # Limit the response to a reasonable length for a name
        temperature=0.2  # Adjust the creativity of the response
    )
    
    # Extract the generated name from the response
    generated_name = response.choices[0].message.content.strip()

    return generated_name

#Label simplification functions
def map_cell_type_labels_to_simplified_set(labels, simplification_level=''):
    """
    Maps a list of labels to a smaller set of labels using the OpenAI API.

    Args:
        labels (list of str): The list of labels to be mapped.
        labels (str): A qualitative description of how much you want the labels to be simplified. Or a direction about how to simplify the labels. Could be anything, like  'extremely', 'barely', 'compartment-level', 'remove-typos'

    Returns:
        dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    #todo, could allow passing custom examples

    #enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the prompt
    labels_str = "    ".join(labels)

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generatory that takes a list of categories and provides a mapping to a {simplification_level} simplified set as a dictionary. The string you return must be valid python and will be directly evaluated as eval(str). Example: Fibroblast    CD8-positive T Cells    CD4-positive T Cells -> {{'Fibroblast':'Fibroblast','CD8-positive T Cells':'T Cell','CD4-positive T Cells':'T Cell'}}"},
        {"role": "user", "content": f"{labels_str} -> "}
    ]

    # Initialize the OpenAI client
    client = get_openai_client()

    # Call the OpenAI Chat Completions API to generate the mapping
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,  # Adjust max tokens as needed
        temperature=0  # Adjust the creativity of the response
    )
    
    # Extract the generated mapping from the response
    generated_mapping = response.choices[0].message.content.strip()

    # Convert the generated mapping to a dictionary
    mapping_dict = eval(generated_mapping)

    return mapping_dict

def map_gene_labels_to_simplified_set(labels, simplification_level=''):
    """
    Maps a list of genes to a smaller set of labels using the OpenAI API.

    Args:
        labels (list of str): The list of labels to be mapped.
        labels (str): A qualitative description of how much you want the labels to be simplified. Could be anything, like  'extremely', 'barely', or 'compartment-level'

    Returns:
        dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    #enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the prompt
    labels_str = "    ".join(labels)

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generatory that takes a list of genes and provides a mapping to a {simplification_level} simplified set as a dictionary. The string you return must be valid python and will be directly evaluated as eval(str). Example: HSP90AA1    HSPA1A    HSPA1B    CLOCK    ARNTL    PER1    IL1A    IL6 -> {{'HSP90AA1':'Heat Shock Proteins','HSPA1A':'Heat Shock Proteins','HSPA1B':'Heat Shock Proteins','CLOCK':'Circadian Rhythm','ARNTL':'Circadian Rhythm','PER1':'Circadian Rhythm','IL1A':'Interleukins','IL6':'Interleukins'}}"},
        {"role": "user", "content": f"{labels_str} -> "}
    ]

    # Initialize the OpenAI client
    client = get_openai_client()

    # Call the OpenAI Chat Completions API to generate the mapping
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,  # Adjust max tokens as needed
        temperature=0  # Adjust the creativity of the response
    )
    
    # Extract the generated mapping from the response
    generated_mapping = response.choices[0].message.content.strip()

    # Convert the generated mapping to a dictionary
    mapping_dict = eval(generated_mapping)

    return mapping_dict

#Biological inference functions
def ai_biological_process(gene_list):
    """
    Describes the most prominent biological process represented by a list of genes using the OpenAI API.

    Args:
        gene_list (list of str): The list of genes to be described.

    Returns:
        dict: A dictionary containing the description of the biological process.
    """
    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        gpt_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list[:-1])
        gpt_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the genes: {genes_str}, and {gene_list[-1]}"

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": gpt_prompt}
    ]

    # Initialize the OpenAI client
    client = get_openai_client()

    # Call the OpenAI Chat Completions API to generate the annotation
    gpt_annotation = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
        temperature=0
    )

    # Extract the generated annotation from the response
    annotation = gpt_annotation.choices[0].message.content.strip()

    return annotation

def ai_cell_type(gene_list):
    """
    Returns the cell type based on a list of marker genes as determined by AI using the OpenAI API.

    Args:
        gene_list (list of str): The list of genes to be described.

    Returns:
        str: The cell type label generate by AI
    """
    #todo
    #add option to pass tissue of origin to prompt

    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        gpt_prompt = f"In a few words and without restating any part of the question, describe the single most likely cell type represented by the marker gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list)
        gpt_prompt = f"In a few words and without restating any part of the question, describe the single most likely cell type represented by the marker genes: {genes_str}"

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": gpt_prompt}
    ]

    # Initialize the OpenAI client
    client = get_openai_client()

    # Call the OpenAI Chat Completions API to generate the annotation
    gpt_annotation = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
        temperature=0
    )

    # Extract the generated annotation from the response
    annotation = gpt_annotation.choices[0].message.content.strip()

    return annotation