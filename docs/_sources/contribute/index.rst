Contribute
===========

Here, we explain the organizing principles of AnnDictionary.

How to contribute
------------------

LLM support
~~~~~~~~~~~~
This package uses LangChain to offer support for almost all mainstream LLM providers.  
To simplify the use of LLMs for the end user, we have wrapped LLM configuration and switching into a single line of code.  
To add support for an LLM, you first need to know how this architecture works.

We defined the class ``LLMManager`` (found in ``anndict/llm/llm_manager.py``) to manage LLM configuration and usage.  
We use the dataclass ``LLMProviders`` to store the configuration procedures for each LLM provider.  
Each provider requires 3 things in their stored configuration procedure:

1. ``class_name``: the class name of the Langchain LLM client for that provider.

2. ``module_path``: the path to the module that contains the class with ``class_name``.

3. ``init_class``: a function that provides the necessary ``constructor_args`` to the langchain llm client class

Here's an example (the entry for OpenAI):

.. code-block:: python

   "openai": LLMProviderConfig(
       class_name="ChatOpenAI",
       module_path="langchain_openai.chat_models",
       init_class=DefaultLLMInitializer,
   )


Wrapping Your Package
~~~~~~~~~~~~~~~~~~~~~~
To wrap a package called "my_package" and include it in Anndictionary as a Wrappers submodule,  
make a file called ``my_package_.py``, and add it to the ``anndict/wrappers`` directory.  
Put all the function wrappers you want into this file, and then access them with:

.. code-block:: python

   from anndict import anndict.wrappers.my_package_
   my_func_wrapper(adata_dict,**kwargs)


Annotation Methods
~~~~~~~~~~~~~~~~~~~
To add an annotation method, first classify it as annotating groups of cells or groups of genes.  
Then, classify the type of annotation: de novo, label transfer, or error correction. Based on these two classifications,  
add your module to the correct subdirectory of ``anndict/annotate``.

Plotting Functions
~~~~~~~~~~~~~~~~~~~
To add a plotting function, create a new file in the ``anndict/plot`` directory. Each file should contain one type of plot.
