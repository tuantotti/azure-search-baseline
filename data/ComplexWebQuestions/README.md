WebComplexQuestions - v1.0.0 - 2018-03-17
----------------------------------------------

This package contains ComplexWebQuestions, a dataset that contains a large set of complex questions in natural language.

(c) 2018.  Alon Talmor, Tel-Aviv University.

LICENSE

The software is licensed under the full GPL v2+.  Please see the file LICENCE.txt

For more information, bug reports, and fixes, contact:
    Alon Talmor
    Dept of Computer Science, Gates 2A
    Stanford CA 94305-9020
    USA
    java-nlp-support@lists.stanford.edu
    http://www-nlp.stanford.edu/software/CRF-NER.shtml

CONTACT

For questions about this distribution, please contact Stanford's JavaNLP group
at java-nlp-user@lists.stanford.edu.  We provide assistance on a best-effort
basis.


QUESTION FILES

The dataset contains 34,689 examples divided into 27,734 train, 3,480 dev, 3,475 test.
each containing:

"ID”: The unique ID of the example; 
"webqsp_ID": The original WebQuestionsSP ID from which the question was constructed; 
"webqsp_question": The WebQuestionsSP Question from which the question was constructed; 
"machine_question": The artificial complex question, before paraphrasing; 
"question": The natural language complex question; 
"sparql": Freebase SPARQL query for the question. Note that the SPARQL was constructed for the machine question, the actual question after paraphrasing
may differ from the SPARQL. 
"compositionality_type": An estimation of the type of compositionally. {composition, conjunction, comparative, superlative}. The estimation has not been manually verified,
 the question after paraphrasing may differ from this estimation.
"answers": a list of answers each containing answer: the actual answer; answer_id: the Freebase answer id; aliases: freebase extracted aliases for the answer.
"created": creation time

NOTE: test set does not contain “answer” field. For test evaluation please send email to 
alontalmor@mail.tau.ac.il.


WEB SNIPPET FILES

"question_ID”: the ID of related question; 
"question": The natural language complex question; 
"web_query": Query sent to the search engine. 
"web_snippets": ~100 web snippets per query. Each snippet includes Title,Snippet.

The snippets file contains 8,488,119 snippets each containing




--------------------
CHANGES
--------------------

2018-03-17      1.0     Initial release

