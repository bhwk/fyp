# Todo

- [x] Figure out better RAG storage structure. Chunks getting cutoff, we're losing context in these nodes.
- [ ] Experiment with intermediary LLM steps to determine the relevance of chunks retrieved, which should reduce the amount of tokens we pass to the final LLM
- [ ] Maybe migrate to a structured DB. Use SQL to retrieve context instead of using Vector embeddings since our data is mostly structured.
- [ ] Seems like 7-8b models fail to make use of tools properly. Tested with Mistral-Nemo-12b Q4, worked a lot better compared to 7B Q8.
