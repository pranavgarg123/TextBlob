.. _extensions:

**********
Extensions
**********

TextBlob supports adding custom models and new languages through "extensions".

Extensions can be installed from the PyPI. ::

    $ pip install textblob-name

where "name" is the name of the package.


Available extensions
====================

Languages
---------

* `textblob-fr <https://github.com/sloria/textblob-fr>`_: French
* `textblob-de <https://github.com/markuskiller/textblob-de>`_: German

Part-of-speech Taggers
----------------------

* `textblob-aptagger <https://github.com/sloria/textblob-aptagger>`_: A fast and accurate tagger based on the Averaged Perceptron.

Transformer-Based Sentiment Analysis
------------------------------------

TextBlob now supports sentiment analysis using pre-trained Transformer models from Hugging Face. To use the `TransformerAnalyzer`, you can specify it when creating a `TextBlob` object:

```python
from textblob import TextBlob
from textblob.sentiments import TransformerAnalyzer

blob = TextBlob("I love this product!", analyzer=TransformerAnalyzer())
print(blob.sentiment)
```

.. admonition:: Interested in creating an extension?

    See the :ref:`Contributing guide <extension-development>`.
